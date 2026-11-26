"""Corporate dark-theme Streamlit web interface for ATS Phase 1.5."""

from __future__ import annotations

import difflib
import html
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import config
from src.models import ExtractionError
from src.ui_helpers import (
    WorkspaceItem,
    add_items_to_workspace,
    compute_cleaning_stats,
    decode_image_bytes,
    export_workspace_csv,
    export_workspace_json,
    find_sample_cv,
    load_benchmark_summary,
    remove_workspace_item,
    render_pdf_preview,
    render_preprocessed_preview,
    reprocess_workspace,
)

logger = logging.getLogger(__name__)

_METHOD_LABELS: dict[str, str] = {
    "direct_pdf": "Direct PDF",
    "ocr_pdf": "OCR PDF",
    "ocr_image": "OCR Image",
    "docx": "Word DOCX",
}

_CORPORATE_CSS = """
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .ats-header-title {
        font-size: 1.65rem;
        font-weight: 600;
        color: #E6EDF3;
        margin-bottom: 0.15rem;
        letter-spacing: -0.01em;
    }
    .ats-header-subtitle {
        font-size: 0.92rem;
        color: #8B949E;
        margin-bottom: 1.1rem;
    }
    .ats-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 0.9rem;
    }
    .ats-card h4 {
        margin: 0 0 0.5rem 0;
        color: #E6EDF3;
        font-size: 1.0rem;
        font-weight: 600;
    }
    .ats-card p, .ats-card li {
        color: #C9D1D9;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .ats-badge {
        display: inline-block;
        background-color: #1F2937;
        border: 1px solid #30363D;
        color: #93C5FD;
        border-radius: 999px;
        padding: 0.18rem 0.62rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        font-size: 0.78rem;
        font-weight: 500;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .ats-badge-row {
        margin: 0.35rem 0 0.75rem 0;
    }
    .ats-muted {
        color: #8B949E;
        font-size: 0.84rem;
    }
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 0.75rem 0.95rem;
    }
</style>
"""


def _init_session_state() -> None:
    """Initialize persistent Streamlit session state keys safely."""
    defaults: dict[str, Any] = {
        "workspace": [],
        "uploader_nonce": 0,
        "selected_hash": None,
        "notices": [],
        "ocr_dpi": config.OCR_DPI,
        "apply_threshold": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _ensure_valid_selection() -> WorkspaceItem | None:
    """Keep ``selected_hash`` synchronized with current workspace contents."""
    workspace: list[WorkspaceItem] = st.session_state["workspace"]
    if not workspace:
        st.session_state["selected_hash"] = None
        return None

    selected_hash = st.session_state.get("selected_hash")
    for item in workspace:
        if item.content_hash == selected_hash:
            return item

    st.session_state["selected_hash"] = workspace[0].content_hash
    return workspace[0]


def _render_sidebar() -> None:
    """Render extraction settings and workspace management controls in the sidebar."""
    with st.sidebar:
        st.markdown("### Extraction Settings")
        dpi_options = [150, 200, 300]
        current_dpi = st.session_state["ocr_dpi"]
        dpi_index = dpi_options.index(current_dpi) if current_dpi in dpi_options else 2

        selected_dpi = st.selectbox(
            "OCR DPI",
            options=dpi_options,
            index=dpi_index,
            help="Resolution used when rasterizing scanned PDF pages for Tesseract OCR.",
        )
        selected_threshold = st.toggle(
            "Adaptive Thresholding",
            value=bool(st.session_state["apply_threshold"]),
            help="Optional Gaussian adaptive binarization for uneven lighting.",
        )
        st.caption("Enable for dark, shadowed, or low-contrast photos")

        st.session_state["ocr_dpi"] = int(selected_dpi)
        st.session_state["apply_threshold"] = bool(selected_threshold)

        workspace: list[WorkspaceItem] = st.session_state["workspace"]
        reprocess_clicked = st.button(
            "Re-process Loaded Files",
            type="primary",
            disabled=len(workspace) == 0,
            width="stretch",
        )
        if reprocess_clicked and workspace:
            with st.spinner("Re-processing workspace files with updated settings..."):
                updated, errors = reprocess_workspace(
                    workspace,
                    ocr_dpi=st.session_state["ocr_dpi"],
                    apply_threshold=st.session_state["apply_threshold"],
                )
            st.session_state["workspace"] = updated
            notices: list[tuple[str, str]] = [
                (
                    "success",
                    f"Re-processed {len(updated)} file(s) at {st.session_state['ocr_dpi']} DPI "
                    f"(Thresholding: {'On' if st.session_state['apply_threshold'] else 'Off'}).",
                )
            ]
            for err in errors:
                notices.append(("error", err))
            st.session_state["notices"] = notices
            st.rerun()

        st.divider()
        st.markdown("### Workspace Summary")
        st.markdown(
            f"- **Loaded CVs:** `{len(workspace)}`\n"
            f"- **Active DPI:** `{st.session_state['ocr_dpi']}`\n"
            f"- **Adaptive Threshold:** `{'Enabled' if st.session_state['apply_threshold'] else 'Disabled'}`"
        )

        if st.button(
            "Clear All Files",
            disabled=len(workspace) == 0,
            width="stretch",
        ):
            st.session_state["workspace"] = []
            st.session_state["selected_hash"] = None
            st.session_state["uploader_nonce"] += 1
            st.session_state["notices"] = [("info", "Workspace cleared.")]
            st.rerun()


def _handle_additive_upload() -> None:
    """Render the non-resetting additive upload widget and process new batches."""
    nonce = st.session_state["uploader_nonce"]
    uploaded_files = st.file_uploader(
        "Add CV File(s)",
        type=["pdf", "png", "jpg", "jpeg", "docx"],
        accept_multiple_files=True,
        key=f"cv_uploader_{nonce}",
        help="Uploaded files are appended to your active workspace and deduplicated by SHA-256 content hash.",
    )
    if not uploaded_files:
        return

    incoming = [(file_obj.name, file_obj.getvalue()) for file_obj in uploaded_files]
    with st.spinner(f"Extracting {len(incoming)} uploaded file(s)..."):
        updated, added, duplicates, errors = add_items_to_workspace(
            st.session_state["workspace"],
            incoming,
            ocr_dpi=st.session_state["ocr_dpi"],
            apply_threshold=st.session_state["apply_threshold"],
        )

    st.session_state["workspace"] = updated
    if added and updated:
        # Focus the most recently added item when starting fresh or adding a single file.
        st.session_state["selected_hash"] = updated[-1].content_hash
    st.session_state["uploader_nonce"] = nonce + 1

    notices: list[tuple[str, str]] = []
    if added:
        notices.append(("success", f"Added {len(added)} file(s): {', '.join(added)}"))
    if duplicates:
        notices.append(
            (
                "info",
                f"Skipped {len(duplicates)} duplicate file(s) already in workspace: "
                f"{', '.join(duplicates)}",
            )
        )
    for err in errors:
        notices.append(("error", err))
    st.session_state["notices"] = notices
    st.rerun()


def _render_notices() -> None:
    """Display transient status notices once and clear them."""
    notices: list[tuple[str, str]] = st.session_state.get("notices", [])
    if not notices:
        return
    for level, message in notices:
        if level == "success":
            st.success(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)
    st.session_state["notices"] = []


def _render_empty_state() -> None:
    """Render the clean zero-file landing card with optional local sample loader."""
    st.markdown(
        """
        <div class="ats-card">
            <h4>Ready for Document Ingestion</h4>
            <p>
                Drag and drop one or more CVs above to begin. Uploads are <strong>additive</strong>:
                you can inspect a single resume and continue adding more files at any time without
                losing existing workspace results.
            </p>
            <ul>
                <li><strong>Supported Formats:</strong> <code>.pdf</code>, <code>.png</code>, <code>.jpg</code>, <code>.jpeg</code>, <code>.docx</code></li>
                <li><strong>Smart Router:</strong> Digital PDFs use direct PyMuPDF extraction (<code>sort=True</code>); scanned PDFs (&lt; 50 non-whitespace chars) and images run through OpenCV preprocessing + Tesseract 5 OCR.</li>
                <li><strong>Technical Term Protection:</strong> Preserves tokens such as <code>C++</code>, <code>C#</code>, <code>.NET</code>, <code>ASP.NET</code>, <code>Node.js</code>, <code>Vue.js</code>, <code>CI/CD</code>, <code>TCP/IP</code>, and <code>PL/SQL</code>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_path = find_sample_cv()
    if sample_path is not None:
        if st.button("Load Sample CV", type="secondary"):
            sample_bytes = sample_path.read_bytes()
            with st.spinner(f"Loading sample CV ({sample_path.name})..."):
                updated, added, duplicates, errors = add_items_to_workspace(
                    st.session_state["workspace"],
                    [(sample_path.name, sample_bytes)],
                    ocr_dpi=st.session_state["ocr_dpi"],
                    apply_threshold=st.session_state["apply_threshold"],
                )
            st.session_state["workspace"] = updated
            if updated:
                st.session_state["selected_hash"] = updated[-1].content_hash
            if added:
                st.session_state["notices"] = [
                    ("success", f"Loaded sample CV: {sample_path.name}")
                ]
            elif duplicates:
                st.session_state["notices"] = [
                    ("info", f"Sample CV {sample_path.name} is already loaded.")
                ]
            elif errors:
                st.session_state["notices"] = [("error", errors[0])]
            st.rerun()


def _render_batch_panel(workspace: list[WorkspaceItem], active_item: WorkspaceItem) -> WorkspaceItem:
    """Render top-level batch summary table, exports, and CV selector for 2+ files."""
    st.markdown("#### Batch Workspace Summary")
    table_rows = [
        {
            "Filename": item.filename,
            "Method": item.result.method,
            "Pages": item.result.page_count,
            "Cleaned Chars": len(item.result.cleaned_text),
            "Time (s)": round(item.result.processing_time_sec, 2),
            "Warnings": "; ".join(item.result.warnings) if item.result.warnings else "None",
        }
        for item in workspace
    ]
    st.dataframe(table_rows, width="stretch", hide_index=True)

    col_sel, col_rm, col_json, col_csv = st.columns([2.4, 0.9, 1.0, 1.0])

    hashes = [item.content_hash for item in workspace]
    labels_by_hash = {
        item.content_hash: f"{item.filename}  [{item.result.method}]"
        for item in workspace
    }
    current_idx = hashes.index(active_item.content_hash)

    with col_sel:
        chosen_hash = st.selectbox(
            "Select CV to Inspect",
            options=hashes,
            index=current_idx,
            format_func=lambda h: labels_by_hash[h],
        )
        st.session_state["selected_hash"] = chosen_hash

    selected_item = next(
        (item for item in workspace if item.content_hash == chosen_hash),
        active_item,
    )

    with col_rm:
        st.markdown("<div style='height: 1.72rem'></div>", unsafe_allow_html=True)
        if st.button("Remove", key="remove_batch_item", width="stretch"):
            st.session_state["workspace"] = remove_workspace_item(
                workspace,
                selected_item.content_hash,
            )
            st.session_state["notices"] = [
                ("info", f"Removed '{selected_item.filename}' from workspace.")
            ]
            st.rerun()

    with col_json:
        st.markdown("<div style='height: 1.72rem'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download All (JSON)",
            data=export_workspace_json(workspace).encode("utf-8"),
            file_name="ats_workspace_results.json",
            mime="application/json",
            width="stretch",
        )

    with col_csv:
        st.markdown("<div style='height: 1.72rem'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download Summary (CSV)",
            data=export_workspace_csv(workspace).encode("utf-8"),
            file_name="ats_workspace_summary.csv",
            mime="text/csv",
            width="stretch",
        )

    st.divider()
    return selected_item


def _render_kpi_strip(item: WorkspaceItem) -> None:
    """Render compact 5-metric KPI bar and prominent warnings banner if needed."""
    res = item.result
    method_label = _METHOD_LABELS.get(res.method, res.method)
    warning_count = len(res.warnings)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Extraction Method", method_label, delta=res.method, delta_color="off")
    c2.metric("Pages", str(res.page_count))
    c3.metric("Characters (Cleaned)", f"{len(res.cleaned_text):,}")
    c4.metric("Processing Time", f"{res.processing_time_sec:.2f} s")
    c5.metric("Warnings", "None" if warning_count == 0 else str(warning_count))

    if res.warnings:
        warning_lines = "\n".join(f"- {msg}" for msg in res.warnings)
        st.warning(f"**Extraction Warnings**\n{warning_lines}")


def _render_document_preview(item: WorkspaceItem) -> None:
    """Render the left-hand Document Preview column with Original/OpenCV toggle."""
    st.markdown("#### Document Preview")
    ext = item.extension

    if ext in config.SUPPORTED_DOCX_EXTENSIONS:
        st.markdown(
            """
            <div class="ats-card">
                <h4>Word Document (.docx)</h4>
                <p>
                    DOCX files are parsed directly from paragraphs and tables without
                    raster rendering. All extracted content is available in the
                    <strong>Extraction Output</strong> panel on the right.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    view_mode = st.segmented_control(
        "Preview Mode",
        options=["Original", "Preprocessed (OpenCV)"],
        default="Original",
        required=True,
        key=f"preview_mode_{item.content_hash}",
        label_visibility="collapsed",
    )
    active_mode = view_mode or "Original"

    try:
        if ext in config.SUPPORTED_PDF_EXTENSIONS:
            page_count = max(1, item.result.page_count)
            selected_page = 1
            if page_count > 1:
                selected_page = st.selectbox(
                    f"Page 1 of {page_count}",
                    options=list(range(1, page_count + 1)),
                    format_func=lambda p: f"Page {p} of {page_count}",
                    key=f"pdf_page_{item.content_hash}",
                    label_visibility="collapsed",
                )
            rgb_image, actual_pages = render_pdf_preview(
                item.file_bytes,
                page_index=selected_page - 1,
                preview_dpi=150,
            )
            base_caption = f"{item.filename} — Page {selected_page} of {actual_pages}"
        else:
            rgb_image = decode_image_bytes(item.file_bytes)
            base_caption = item.filename

        if active_mode == "Preprocessed (OpenCV)":
            use_threshold = bool(st.session_state["apply_threshold"])
            preview_img = render_preprocessed_preview(
                rgb_image,
                apply_threshold=use_threshold,
            )
            threshold_note = " + adaptive threshold" if use_threshold else ""
            st.image(
                preview_img,
                caption=f"{base_caption} (OpenCV: grayscale, denoise, conditional deskew{threshold_note})",
                width="stretch",
            )
        else:
            st.image(rgb_image, caption=base_caption, width="stretch")
    except ExtractionError as exc:
        st.error(f"Preview unavailable: {exc}")


def _render_extraction_output(item: WorkspaceItem) -> None:
    """Render the right-hand Extraction Output column with 3 structured sub-tabs."""
    st.markdown("#### Extraction Output")
    res = item.result
    stats = compute_cleaning_stats(res.raw_text, res.cleaned_text)
    stem = Path(item.filename).stem

    tab_clean, tab_diff, tab_json = st.tabs(
        ["Cleaned Text", "Raw vs. Cleaned", "JSON / Metadata"]
    )

    with tab_clean:
        if stats.preserved_tech_terms:
            badges_html = "".join(
                f'<span class="ats-badge">{html.escape(term)}</span>'
                for term in stats.preserved_tech_terms
            )
            st.markdown(
                f'<div class="ats-muted">Preserved Technical Terms</div>'
                f'<div class="ats-badge-row">{badges_html}</div>',
                unsafe_allow_html=True,
            )

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Download .txt",
                data=res.cleaned_text.encode("utf-8"),
                file_name=f"{stem}_cleaned.txt",
                mime="text/plain; charset=utf-8",
                key=f"dl_txt_{item.content_hash}",
                width="stretch",
            )
        with dl_col2:
            st.download_button(
                "Download .json",
                data=json.dumps(res.to_dict(), ensure_ascii=False, indent=2).encode(
                    "utf-8"
                ),
                file_name=f"{stem}_result.json",
                mime="application/json",
                key=f"dl_json_{item.content_hash}",
                width="stretch",
            )

        with st.container(height=500, border=True):
            st.code(
                res.cleaned_text if res.cleaned_text else "(No readable text extracted)",
                language=None,
            )

    with tab_diff:
        s1, s2, s3 = st.columns(3)
        s1.metric("Raw Characters", f"{stats.raw_chars:,}")
        s2.metric("Cleaned Characters", f"{stats.cleaned_chars:,}")
        s3.metric("Noise Removed", f"{stats.reduction_pct:.2f}%")

        compare_mode = st.segmented_control(
            "Comparison Mode",
            options=["Side-by-Side", "Unified Diff"],
            default="Side-by-Side",
            required=True,
            key=f"compare_mode_{item.content_hash}",
            label_visibility="collapsed",
        )
        if (compare_mode or "Side-by-Side") == "Side-by-Side":
            raw_col, clean_col = st.columns(2)
            with raw_col:
                st.caption("Raw Extracted Text")
                with st.container(height=420, border=True):
                    st.code(res.raw_text or "(Empty)", language=None)
            with clean_col:
                st.caption("Cleaned Text")
                with st.container(height=420, border=True):
                    st.code(res.cleaned_text or "(Empty)", language=None)
        else:
            diff_lines = list(
                difflib.unified_diff(
                    res.raw_text.splitlines(),
                    res.cleaned_text.splitlines(),
                    fromfile="raw_text",
                    tofile="cleaned_text",
                    lineterm="",
                )
            )
            diff_text = (
                "\n".join(diff_lines)
                if diff_lines
                else "No textual differences between raw and cleaned output."
            )
            with st.container(height=420, border=True):
                st.code(diff_text, language="diff")

    with tab_json:
        with st.container(height=540, border=True):
            st.json(res.to_dict())


def _render_inspector_tab() -> None:
    """Render Top Tab 1: CV Ingestion & Inspector."""
    _handle_additive_upload()
    _render_notices()

    workspace: list[WorkspaceItem] = st.session_state["workspace"]
    active_item = _ensure_valid_selection()

    if not workspace or active_item is None:
        _render_empty_state()
        return

    if len(workspace) == 1:
        title_col, action_col = st.columns([5, 1])
        with title_col:
            st.markdown(f"### Inspecting: `{active_item.filename}`")
        with action_col:
            if st.button("Remove File", key="remove_single_item", width="stretch"):
                st.session_state["workspace"] = remove_workspace_item(
                    workspace,
                    active_item.content_hash,
                )
                st.session_state["notices"] = [
                    ("info", f"Removed '{active_item.filename}' from workspace.")
                ]
                st.rerun()
    else:
        active_item = _render_batch_panel(workspace, active_item)

    _render_kpi_strip(active_item)
    st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.0, 1.15], gap="large")
    with left_col:
        _render_document_preview(active_item)
    with right_col:
        _render_extraction_output(active_item)


def _render_job_matching_tab() -> None:
    """Render Top Tab 2: Phase 2 Job Matching Scaffold."""
    workspace: list[WorkspaceItem] = st.session_state["workspace"]
    candidate_count = len(workspace)

    st.info(f"**{candidate_count} candidate(s) currently loaded in workspace**")

    st.text_input(
        "Target Job Title",
        placeholder="e.g., Senior Python Backend Engineer",
        key="phase2_job_title",
    )
    st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the full job requirements, preferred skills, and role responsibilities...",
        key="phase2_job_description",
    )

    st.button(
        "Analyze & Rank Candidates (Phase 2)",
        type="primary",
        disabled=True,
        width="stretch",
        help="Available in Phase 2 once local NLP and ranking engines are integrated.",
    )

    st.markdown(
        """
        <div class="ats-card">
            <h4>Phase 2 Matching Architecture (Upcoming)</h4>
            <p>
                This workspace connects directly to the Phase 1 <code>ExtractionResult</code>
                schema without reprocessing uploaded CVs. Planned local NLP components:
            </p>
            <ul>
                <li><strong>Local spaCy Skill Extraction:</strong> Entity and phrase matching for technical competencies, degrees, and experience entities without external API calls.</li>
                <li><strong>TF-IDF + Cosine Similarity Scoring:</strong> Transparent, reproducible relevance ranking across all candidates loaded in the workspace.</li>
                <li><strong>Missing Skill Gap Analysis:</strong> Per-candidate breakdown of matched requirements versus missing job description skills.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_benchmark_tab() -> None:
    """Render Top Tab 3: OCR Benchmark Metrics Viewer."""
    summary = load_benchmark_summary(config.BENCHMARK_OUTPUT_PATH)
    if summary is None:
        st.info(
            "No OCR benchmark report found at `outputs/ocr_benchmark.csv`. "
            "Run the evaluator from your terminal to generate reproducible Ground Truth metrics:"
        )
        st.code("python -m src.evaluator --sample-size 10", language="bash")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall WER", f"{summary.overall_wer:.2%}")
    k2.metric("Overall CER", f"{summary.overall_cer:.2%}")
    k3.metric("Avg Time (s)", f"{summary.avg_time_sec:.2f} s")
    k4.metric("Evaluated Samples", str(summary.total_samples))

    st.markdown("#### Clean vs. Degraded Performance")
    group_rows = [
        {
            "Group": group_name,
            "Samples": int(metrics["count"]),
            "Mean WER": float(metrics["mean_wer"]),
            "Mean CER": float(metrics["mean_cer"]),
            "Mean Time (s)": float(metrics["mean_time_sec"]),
        }
        for group_name, metrics in summary.group_stats.items()
    ]

    tbl_col, chart_col = st.columns([1.05, 1.0], gap="large")
    with tbl_col:
        st.dataframe(group_rows, width="stretch", hide_index=True)
    with chart_col:
        chart_df = pd.DataFrame(group_rows).set_index("Group")[["Mean WER", "Mean CER"]]
        st.bar_chart(chart_df, width="stretch")

    st.markdown("#### Per-Sample Benchmark Results")
    st.dataframe(summary.rows, width="stretch", hide_index=True)


def main() -> None:
    """Configure page layout, apply corporate theme styles, and render UI tabs."""
    st.set_page_config(
        page_title="ATS Resume Parser & Inspector",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CORPORATE_CSS, unsafe_allow_html=True)
    _init_session_state()
    _render_sidebar()

    st.markdown(
        '<div class="ats-header-title">ATS Resume Parser &amp; Inspector</div>'
        '<div class="ats-header-subtitle">'
        "Local Open-Source CV Ingestion, Smart-Routing OCR &amp; Text Quality Workbench"
        "</div>",
        unsafe_allow_html=True,
    )

    tab_inspector, tab_matching, tab_benchmark = st.tabs(
        [
            "CV Ingestion & Inspector",
            "Job Matching (Phase 2)",
            "OCR Benchmark",
        ]
    )

    with tab_inspector:
        _render_inspector_tab()
    with tab_matching:
        _render_job_matching_tab()
    with tab_benchmark:
        _render_benchmark_tab()


if __name__ == "__main__":
    main()
