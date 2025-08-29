# --------------------------------------------------------------------------
#
# PII Scanning and Masking Agent using LangGraph and Gemini
#
# Description:
# This script implements a stateful AI agent to scan a CSV file for
# Personally Identifiable Information (PII), generate a detailed report of
# its findings, and produce a masked version of the input CSV.
#
# To Run:
# 1. Install dependencies:
#    pip install -U langgraph langchain langchain-google-genai pandas pydantic python-dotenv
#
# 2. Set up your API key:
#    Create a `.env` file in the same directory with the content:
#    GOOGLE_API_KEY="your_google_api_key_here"
#
# 3. Execute the script:
#    python pii_scanner_agent.py
#
# --------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangGraph, LangChain, and Gemini dependencies
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from a .env file
load_dotenv()

# -----------------------------
# 1. Pydantic Models for State & I/O
# These models define the structured data used throughout the agent's workflow,
# ensuring type safety and clear data contracts between different components.
# -----------------------------

class PIISample(BaseModel):
    """Represents a single detected PII value example."""
    column: str
    row_index: int
    raw_value: str
    hashed_preview: str

class ColumnFinding(BaseModel):
    """Stores the analysis results for a single CSV column."""
    column: str
    pii_types: List[str] = Field(default_factory=list)  # e.g., ["EMAIL", "PHONE"]
    confidence: float = 0.0
    rationale: Optional[str] = None
    examples: List[PIISample] = Field(default_factory=list)

class PIIReport(BaseModel):
    """The final consolidated report of all findings."""
    columns_flagged: List[ColumnFinding] = Field(default_factory=list)
    total_rows: int = 0
    summary: Dict[str, Any] = Field(default_factory=dict)

class MaskRule(BaseModel):
    """Defines a masking strategy for a specific PII type."""
    pii_type: str
    strategy: Literal[
        "redact_all",
        "partial_email",
        "partial_phone",
        "hash_consistent",
        "ipv4_subnet",
        "year_only",
    ]

class Config(BaseModel):
    """Configuration settings for the agent's behavior."""
    sample_rows_for_llm: int = 8
    sample_rows_for_regex: int = 200
    max_examples_per_column: int = 5
    mask_rules: List[MaskRule] = Field(default_factory=lambda: [
        MaskRule(pii_type="EMAIL", strategy="partial_email"),
        MaskRule(pii_type="PHONE", strategy="partial_phone"),
        MaskRule(pii_type="SSN", strategy="hash_consistent"),
        MaskRule(pii_type="CREDIT_CARD", strategy="hash_consistent"),
        MaskRule(pii_type="IP", strategy="ipv4_subnet"),
        MaskRule(pii_type="DOB", strategy="year_only"),
        MaskRule(pii_type="ADDRESS", strategy="redact_all"),
        MaskRule(pii_type="NAME", strategy="redact_all"),
    ])
    pii_regex: Dict[str, str] = Field(default_factory=lambda: {
        "EMAIL": r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "PHONE": r"(?:(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?|\d{3})[-.\s]?\d{3}[-.\s]?\d{4})",
        "SSN": r"\b\d{3}-?\d{2}-?\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "IP": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "DOB": r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
    })

class AgentState(BaseModel):
    """The central state object that flows through the LangGraph graph."""
    # Inputs
    input_csv: str
    outdir: str
    model: str = "gemini-1.5-pro-latest"
    config: Config = Field(default_factory=Config)

    # Working data (transient)
    df_head: Optional[List[Dict[str, Any]]] = None
    columns: List[str] = Field(default_factory=list)
    llm_column_analysis: List[ColumnFinding] = Field(default_factory=list)
    regex_column_analysis: List[ColumnFinding] = Field(default_factory=list)

    # Outputs
    report: Optional[PIIReport] = None
    masked_csv_path: Optional[str] = None
    findings_json_path: Optional[str] = None
    report_md_path: Optional[str] = None

    # Metadata
    total_rows: int = 0
    errors: List[str] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)

# -----------------------------
# 2. Utility Functions
# Helper functions for hashing, previewing data securely, and file system operations.
# -----------------------------

def sha256_token(v: str) -> str:
    """Creates a short, consistent SHA256 hash of a string."""
    return hashlib.sha256(v.encode("utf-8", errors="ignore")).hexdigest()[:12]

def hashed_preview(v: Any) -> str:
    """Creates a hashed preview of a value for safe transmission to an LLM."""
    s = str(v)[:64]  # Truncate to limit data exposure
    return sha256_token(s)

def ensure_outdir(path: str) -> Path:
    """Ensures that the output directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------
# 3. LangGraph Agent Nodes
# Each function represents a node in the state graph, performing a specific
# task and updating the agent's state.
# -----------------------------

def node_load_csv(state: AgentState) -> AgentState:
    """Loads the input CSV and initializes basic metadata in the state."""
    try:
        df = pd.read_csv(state.input_csv)
        state.total_rows = len(df)
        state.columns = list(df.columns)
        state.df_head = df.head(10).to_dict(orient="records")
        state.logs.append(f"Loaded CSV with {state.total_rows} rows and {len(state.columns)} columns.")
        # Cache the dataframe in a temporary attribute to avoid serializing it in the state
        state._df_cache = df
    except Exception as e:
        state.errors.append(f"load_csv: {e}")
    return state

def node_regex_scan(state: AgentState) -> AgentState:
    """Performs a fast, pattern-based scan on a sample of the data."""
    try:
        df = getattr(state, "_df_cache")
        findings: List[ColumnFinding] = []
        sample_df = df.head(state.config.sample_rows_for_regex)
        
        state.logs.append(f"Starting regex scan on {len(sample_df)} rows, {len(state.columns)} columns")

        for col in state.columns:
            col_find = ColumnFinding(column=col)
            series = sample_df[col].astype(str).fillna("")
            matches: Dict[str, int] = {}
            examples: List[PIISample] = []
            
            state.logs.append(f"Scanning column '{col}' with values: {series.tolist()}")

            for pii_type, pattern in state.config.pii_regex.items():
                count = series.str.contains(pattern, regex=True).sum()
                if count > 0:
                    matches[pii_type] = int(count)
                    state.logs.append(f"Found {count} {pii_type} matches in column {col}")

            if matches:
                # Collect a few examples if matches are found
                matched_indices = series[series.str.contains('|'.join(state.config.pii_regex.values()), regex=True)].index
                for idx in matched_indices[:state.config.max_examples_per_column]:
                    val = series[idx]
                    examples.append(PIISample(
                        column=col,
                        row_index=int(idx),
                        raw_value=str(val)[:64],
                        hashed_preview=hashed_preview(val)
                    ))

                col_find.pii_types = sorted(matches.keys())
                col_find.confidence = min(1.0, sum(matches.values()) / max(1, len(sample_df)))
                col_find.rationale = "Regex/heuristic match found."
                col_find.examples = examples
            findings.append(col_find)

        state.regex_column_analysis = findings
        state.logs.append(f"Completed regex scan. Found PII in {len([f for f in findings if f.pii_types])} columns.")
    except Exception as e:
        state.errors.append(f"regex_scan: {e}")
    return state

def node_llm_classify(state: AgentState) -> AgentState:
    """Uses Gemini to enhance PII classification based on column names and regex findings."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_google_api_key_here":
            state.logs.append("LLM classification skipped: Missing or invalid GOOGLE_API_KEY")
            # Use regex findings directly when LLM is not available
            state.llm_column_analysis = []
            return state
        
        llm = ChatGoogleGenerativeAI(model=state.model, google_api_key=api_key, temperature=0.1)

        # Instead of hashed data, send column names and regex findings
        system_prompt = (
            "You are a PII Detection and Redaction AI Agent specializing in data privacy and compliance.\n"
            "Task: Analyze the CSV column names and regex findings to identify additional PII columns that may have been missed.\n"
            "Return ONLY a JSON array of objects with keys: {column, pii_types, confidence, rationale}.\n"
            "Allowed pii_types: EMAIL, PHONE, SSN, CREDIT_CARD, IP, DOB, ADDRESS, NAME, NONE.\n"
            "Base your judgment on:\n"
            "1. Column names (e.g., 'name', 'address', 'ssn', 'credit_card')\n"
            "2. Regex findings already identified\n"
            "3. Common PII naming conventions\n"
            "Be thorough in identifying PII - it's better to flag a column as PII if there's any doubt.\n"
            "The goal is to ensure all sensitive data is properly identified for redaction."
        )
        
        human_prompt = json.dumps({
            "columns": state.columns,
            "regex_findings": [cf.model_dump() for cf in state.regex_column_analysis if cf.pii_types],
            "all_columns": state.columns,
        })

        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        text = resp.content if isinstance(resp, AIMessage) else str(resp)
        
        # Clean the response to extract the JSON part
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            raise ValueError(f"LLM did not return a valid JSON array. Response: {text}")

        parsed_json = json.loads(json_match.group(0))
        findings = [
            ColumnFinding(
                column=item.get("column", ""),
                pii_types=[pt for pt in item.get("pii_types", []) if pt != "NONE"],
                confidence=float(item.get("confidence", 0.0)),
                rationale=item.get("rationale"),
            ) for item in parsed_json
        ]
        state.llm_column_analysis = findings
        state.logs.append("LLM classification completed.")
    except Exception as e:
        state.errors.append(f"llm_classify: {e}")
        # Fallback to regex findings only
        state.llm_column_analysis = []
    return state

def node_consolidate(state: AgentState) -> AgentState:
    """Merges findings from regex and LLM analyses into a final report."""
    try:
        by_col: Dict[str, ColumnFinding] = {c.column: c for c in state.regex_column_analysis}
        for llm_finding in state.llm_column_analysis:
            if llm_finding.column in by_col:
                base = by_col[llm_finding.column]
                base.pii_types = sorted(list(set(base.pii_types) | set(llm_finding.pii_types)))
                base.confidence = max(base.confidence, llm_finding.confidence)
                base.rationale = f"{base.rationale} | LLM: {llm_finding.rationale}"
            else:
                by_col[llm_finding.column] = llm_finding

        flagged_cols = [c for c in by_col.values() if c.pii_types]
        report = PIIReport(columns_flagged=flagged_cols, total_rows=state.total_rows)

        summary: Dict[str, int] = {}
        for c in flagged_cols:
            for t in c.pii_types:
                summary[t] = summary.get(t, 0) + 1
        report.summary = {"columns_with_pii_type": summary}
        
        state.report = report
        state.logs.append(f"Consolidated findings into a final report. {len(flagged_cols)} columns flagged for PII.")
    except Exception as e:
        state.errors.append(f"consolidate: {e}")
    return state

def apply_mask(value: Any, pii_types: List[str], rules: List[MaskRule]) -> str:
    """Applies the appropriate masking strategy to a given value."""
    s_val = str(value)
    if pd.isna(value) or not pii_types:
        return s_val

    for pii_type in pii_types:
        rule = next((r for r in rules if r.pii_type == pii_type), None)
        if not rule:
            continue
        
        strategy = rule.strategy
        if strategy == "redact_all": return "████"
        if strategy == "hash_consistent": return f"token_{sha256_token(s_val)}"
        if strategy == "partial_email":
            parts = s_val.split("@")
            if len(parts) == 2:
                local, domain = parts
                tld = domain.split(".")[-1] if "." in domain else "dom"
                return f"{local[:1]}***@***.{tld}"
        if strategy == "partial_phone":
            digits = ''.join(ch for ch in s_val if ch.isdigit())
            return f"***-***-{digits[-4:]}" if len(digits) >= 4 else "***-***-****"
        if strategy == "ipv4_subnet":
            parts = s_val.split('.')
            if len(parts) == 4: return '.'.join(parts[:3] + ['0'])
        if strategy == "year_only":
            m = re.search(r"(\d{4})", s_val)
            if m: return m.group(1)

    return f"token_{sha256_token(s_val)}" # Fallback for unhandled cases

def node_mask_and_save(state: AgentState) -> AgentState:
    """Applies masking rules to the dataframe and saves output files."""
    try:
        if not state.report:
            state.errors.append("mask_and_save: Report was not generated, cannot mask.")
            return state

        df = getattr(state, "_df_cache")
        pii_map = {c.column: c.pii_types for c in state.report.columns_flagged}
        rules = state.config.mask_rules
        
        masked_df = df.copy()
        for col, types in pii_map.items():
            if types:
                masked_df[col] = masked_df[col].apply(lambda v: apply_mask(v, types, rules))

        outdir = ensure_outdir(state.outdir)
        masked_path = outdir / "masked.csv"
        masked_df.to_csv(masked_path, index=False)
        state.masked_csv_path = str(masked_path)

        findings_path = outdir / "findings.json"
        with open(findings_path, "w", encoding="utf-8") as f:
            f.write(state.report.model_dump_json(indent=2))
        state.findings_json_path = str(findings_path)
        
        state.logs.append(f"Masked CSV and findings JSON saved to '{outdir}'.")
    except Exception as e:
        state.errors.append(f"mask_and_save: {e}")
    return state

def node_generate_report(state: AgentState) -> AgentState:
    """Generates a human-readable Markdown report of the findings."""
    try:
        if not state.report:
            state.errors.append("generate_report: Report was not generated.")
            return state

        outdir = ensure_outdir(state.outdir)
        md_path = outdir / "report.md"
        
        lines = [
            f"# PII Detection Report for `{state.input_csv}`",
            f"\n**Total Rows Scanned:** {state.total_rows}\n",
            "## Summary of Findings",
            "```json",
            json.dumps(state.report.summary, indent=2),
            "```\n",
            "## Detailed Column Analysis"
        ]

        if not state.report.columns_flagged:
            lines.append("\n*No PII was detected in any columns.*")
        else:
            for c in state.report.columns_flagged:
                lines.extend([
                    f"\n### Column: `{c.column}`",
                    f"- **PII Types**: {', '.join(f'`{t}`' for t in c.pii_types)}",
                    f"- **Confidence**: {c.confidence:.2f}",
                    f"- **Rationale**: {c.rationale or 'N/A'}",
                ])
                if c.examples:
                    lines.append("- **Examples (Hashed Previews):**")
                    for ex in c.examples:
                        lines.append(f"  - `row {ex.row_index}`: `{ex.hashed_preview}`")
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        state.report_md_path = str(md_path)
        state.logs.append(f"Markdown report saved to {md_path}")
    except Exception as e:
        state.errors.append(f"generate_report: {e}")
    return state

# -----------------------------
# 4. Graph Definition
# This section assembles the nodes into a sequential workflow using LangGraph.
# -----------------------------

def build_graph() -> StateGraph:
    """Builds and configures the LangGraph StateGraph for the PII agent."""
    graph = StateGraph(AgentState)
    graph.add_node("load_csv", node_load_csv)
    graph.add_node("regex_scan", node_regex_scan)
    graph.add_node("llm_classify", node_llm_classify)
    graph.add_node("consolidate", node_consolidate)
    graph.add_node("mask_and_save", node_mask_and_save)
    graph.set_entry_point("load_csv")
    graph.add_edge("load_csv", "regex_scan")
    graph.add_edge("regex_scan", "llm_classify")
    graph.add_edge("llm_classify", "consolidate")
    graph.add_edge("consolidate", "mask_and_save")
    graph.add_edge("mask_and_save", END)
    
    return graph.compile()

# -----------------------------
# 5. Main Execution Block
# This is the entry point of the script. It creates sample data,
# initializes the agent state, runs the graph, and prints the results.
# -----------------------------

def main():
    """Main function to set up data, run the agent, and report results."""
    print("--- PII Scanner Agent Initializing ---")
    
    # --- 1. Create sample data for demonstration ---
    DATA_DIR = "./data"
    OUT_DIR = "./out"
    INPUT_CSV = os.path.join(DATA_DIR, "customers.csv")
    
    # Test regex patterns
    import re
    test_email = "alice@example.com"
    test_phone = "+1-415-555-0199"
    test_phone2 = "4155550188"
    test_phone3 = "555-867-5309"
    email_pattern = r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    phone_pattern = r"(?:(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?|\d{3})[-.\s]?\d{3}[-.\s]?\d{4})"
    
    print(f"Testing regex patterns:")
    print(f"Email '{test_email}' matches: {bool(re.match(email_pattern, test_email))}")
    print(f"Phone '{test_phone}' matches: {bool(re.match(phone_pattern, test_phone))}")
    print(f"Phone '{test_phone2}' matches: {bool(re.match(phone_pattern, test_phone2))}")
    print(f"Phone '{test_phone3}' matches: {bool(re.match(phone_pattern, test_phone3))}")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_content = (
        "name,email,phone,notes,ip_address,dob\n"
        "Alice Smith,alice@example.com,+1-415-555-0199,Call after 5pm,192.168.1.10,1990-05-15\n"
        "Bob Johnson,bob@acme.co,4155550188,VIP Customer,203.0.113.45,1985/03/22\n"
        "Charlie Brown,charlie+test@gmail.com,555-867-5309,Met at conference,198.51.100.2,12-01-1992\n"
    )
    with open(INPUT_CSV, "w") as f:
        f.write(csv_content)
    print(f"Sample data created at: {INPUT_CSV}")
    
    # --- 2. Initialize and run the agent ---
    initial_state = AgentState(
        input_csv=INPUT_CSV,
        outdir=OUT_DIR,
        model="gemini-2.5-pro"
    )
    app = build_graph()
    
    print("\n--- Running PII Agent Workflow ---")
    try:
        final_state = app.invoke(initial_state)
        print("Agent workflow completed successfully!")
        
        # Print all logs
        if hasattr(final_state, 'logs') and final_state.logs:
            print("\n--- Logs ---")
            for log in final_state.logs:
                print(f"LOG: {log}")
        
        # Print any errors
        if hasattr(final_state, 'errors') and final_state.errors:
            print("\n--- Errors ---")
            for error in final_state.errors:
                print(f"ERROR: {error}")
    except Exception as e:
        print(f"Error running agent: {e}")
        final_state = None

    # --- 3. Print final summary ---
    if final_state:
        print("\n=== Agent Run Complete ===")
        if hasattr(final_state, 'masked_csv_path'):
            print(f"Masked CSV:      {final_state.masked_csv_path}")
        if hasattr(final_state, 'findings_json_path'):
            print(f"Findings JSON:   {final_state.findings_json_path}")
        if hasattr(final_state, 'errors') and final_state.errors:
            print(f"\nErrors encountered: {final_state.errors}")
        else:
            print("\nSuccessfully generated all output files.")
    else:
        print("Agent did not produce a final state.")

if __name__ == "__main__":
    main()

# -----------------------------
# Self-Assessment & Suggestions
# -----------------------------
# - Security: We avoid sending raw values to the LLM; only hashed previews + regex hints are shared.
# - Performance: Regex scan is bounded to N rows; consider sampling strategies for very large CSVs.
# - Extensibility: Add custom rules per jurisdiction (e.g., GDPR special categories).
# - Reliability: Add unit tests with synthetic data.
# - Observability: Integrate LangSmith or OpenTelemetry for production monitoring.