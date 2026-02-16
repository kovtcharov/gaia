# Multi-Document Synthesis Architecture for GAIA

**Date**: February 7, 2026
**Version**: 1.0
**Status**: Specification
**Priority**: HIGH
**Estimated Effort**: 8-10 weeks (2 engineers)
**Target**: Enable cross-document analysis, citation tracking, table extraction, and evidence synthesis

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Architecture Overview](#architecture-overview)
4. [Component Specifications](#component-specifications)
5. [Data Models and Schemas](#data-models-and-schemas)
6. [Document Ingestion Pipeline](#document-ingestion-pipeline)
7. [Cross-Document Analysis Engine](#cross-document-analysis-engine)
8. [Citation Tracking System](#citation-tracking-system)
9. [Table Extraction and Comparison](#table-extraction-and-comparison)
10. [Integration with GAIA](#integration-with-gaia)
11. [Safety and Security Considerations](#safety-and-security-considerations)
12. [Implementation Plan](#implementation-plan)
13. [Testing Strategy](#testing-strategy)
14. [Success Metrics](#success-metrics)
15. [Complete Code](#complete-code)

---

## Executive Summary

### The Gap

GAIA's current RAG system can:
- Index and search individual documents
- Answer questions about single document content
- Parse PDF text and embed chunks

GAIA's RAG system **cannot**:
- Synthesize information across multiple documents simultaneously
- Track citations back to specific pages and paragraphs
- Extract and compare tables across documents
- Detect contradictions or agreements between sources
- Generate evidence-backed reports with proper attribution
- Handle structured data (tables, charts, forms) within documents

### The Solution

A **Multi-Document Synthesis Architecture** that enables:
- Parallel ingestion and indexing of document collections
- Cross-document entity resolution and linking
- Citation-tracked answers with page/paragraph references
- Table extraction, normalization, and cross-document comparison
- Contradiction detection and consensus analysis
- Evidence synthesis reports with confidence scoring
- Integration with GAIA's existing RAG pipeline

### Impact

**Unlocks entire category**: Research and analysis automation for:
- Legal document review (compare contracts, find inconsistencies)
- Academic research synthesis (literature review, citation management)
- Financial analysis (compare reports, extract key metrics)
- Compliance auditing (cross-reference policies with regulations)
- Due diligence (analyze multiple company documents)

**Before**: 30% coverage for multi-document analysis category
**After**: 85% coverage

---

## Problem Statement

### Current Limitations

**Example task**: "Compare the financial projections in Q3 and Q4 reports and identify any inconsistencies"

**What GAIA can do today**:
```python
# Single document RAG only
from gaia.rag.sdk import RAGSDK

rag = RAGSDK()
rag.add_document("q3_report.pdf")
result = rag.query("What are the financial projections?")
# Returns chunks from ONE document, no cross-document analysis
```

**What GAIA needs to do**:
- Ingest both Q3 and Q4 reports simultaneously
- Extract financial tables from each document
- Normalize table structures for comparison
- Identify overlapping metrics and time periods
- Detect discrepancies in projected vs actual figures
- Generate a comparison report with page-level citations
- Highlight contradictions with confidence scores

### User Stories

**Story 1: Legal Contract Comparison**
```
As a legal analyst, I want my agent to:
- Ingest 3 versions of a contract (v1, v2, v3)
- Identify all clauses that changed between versions
- Track which sections were added, modified, or removed
- Generate a redline summary with specific page references
So that I can quickly review contract changes
```

**Story 2: Research Literature Review**
```
As a researcher, I want my agent to:
- Ingest 20 academic papers on a specific topic
- Identify key findings across all papers
- Detect where authors agree or disagree
- Generate a synthesis report with proper citations
- Create a bibliography in standard format
So that I can write a comprehensive literature review
```

**Story 3: Financial Report Analysis**
```
As a financial analyst, I want my agent to:
- Ingest quarterly reports from 5 companies
- Extract revenue, profit, and growth tables
- Normalize metrics for cross-company comparison
- Generate a competitive analysis report
- Include source citations for every data point
So that I can present accurate market analysis
```

**Story 4: Compliance Cross-Reference**
```
As a compliance officer, I want my agent to:
- Ingest company policies and regulatory requirements
- Map each policy section to applicable regulations
- Identify gaps where regulations are not covered
- Generate a compliance matrix with references
So that I can ensure full regulatory compliance
```

---

## Architecture Overview

### High-Level Design

```
+------------------------------------------------------------------+
|                  Document Collection                              |
|  [Doc A]  [Doc B]  [Doc C]  [Doc D]  [Doc E]                    |
+-----+--------+--------+--------+--------+-----------------------+
      |        |        |        |        |
      v        v        v        v        v
+------------------------------------------------------------------+
|                  Ingestion Pipeline                               |
|                                                                   |
|  +------------+  +------------+  +-------------+  +------------+ |
|  | PDF Parser |  | Table      |  | Structure   |  | Metadata   | |
|  | (text,     |  | Extractor  |  | Analyzer    |  | Extractor  | |
|  |  images)   |  | (cells,    |  | (sections,  |  | (title,    | |
|  |            |  |  headers)  |  |  hierarchy) |  |  author)   | |
|  +-----+------+  +-----+------+  +------+------+  +-----+------+ |
|        |              |                |                |         |
|        v              v                v                v         |
|  +-----------------------------------------------------------+   |
|  |              Document Store (per-document index)           |   |
|  |  - Chunks with embeddings and position metadata            |   |
|  |  - Tables with normalized structure                        |   |
|  |  - Section hierarchy with page references                  |   |
|  +-----------------------------------------------------------+   |
+--------------------------------+---------------------------------+
                                 |
                                 v
+------------------------------------------------------------------+
|                Cross-Document Analysis Engine                     |
|                                                                   |
|  +----------------+  +------------------+  +-------------------+ |
|  | Entity         |  | Contradiction    |  | Synthesis         | |
|  | Resolution     |  | Detection        |  | Engine            | |
|  |                |  |                  |  |                   | |
|  | - Name linking |  | - Claim extract  |  | - Multi-source    | |
|  | - Concept map  |  | - Comparison     |  |   evidence        | |
|  | - Reference    |  | - Confidence     |  | - Weighted        | |
|  |   matching     |  |   scoring        |  |   consensus       | |
|  +-------+--------+  +--------+---------+  +--------+----------+ |
|          |                     |                      |           |
|          v                     v                      v           |
|  +-----------------------------------------------------------+   |
|  |              Citation Tracking System                      |   |
|  |  - Page-level references                                   |   |
|  |  - Paragraph-level citations                               |   |
|  |  - Table cell-level references                             |   |
|  |  - Confidence scores per citation                          |   |
|  +-----------------------------------------------------------+   |
+------------------------------------------------------------------+
```

### Component Layers

| Layer | Components | Responsibility |
|-------|-----------|----------------|
| **Ingestion** | PDFParser, TableExtractor, StructureAnalyzer | Document parsing and indexing |
| **Storage** | DocumentStore, ChunkIndex, TableStore | Multi-document persistence |
| **Analysis** | EntityResolver, ContradictionDetector, SynthesisEngine | Cross-document intelligence |
| **Citation** | CitationTracker, ReferenceFormatter, EvidenceChain | Source tracking and attribution |
| **Output** | ReportGenerator, ComparisonRenderer, MatrixBuilder | Result presentation |

---

## Component Specifications

### 1. Document Ingestion Pipeline

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Multi-document ingestion pipeline with structure preservation.
"""

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gaia.logger import get_logger

log = get_logger(__name__)


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "markdown"
    HTML = "html"
    CSV = "csv"
    XLSX = "xlsx"


@dataclass
class PageContent:
    """Content of a single page with position metadata."""
    page_number: int
    text: str
    tables: List["ExtractedTable"] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    bounding_boxes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSection:
    """A logical section within a document."""
    section_id: str
    title: str
    level: int  # Heading level (1 = H1, 2 = H2, etc.)
    content: str
    page_start: int
    page_end: int
    parent_section_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    char_start: int = 0  # Character offset in full text
    char_end: int = 0


@dataclass
class DocumentChunk:
    """A chunk of text with full provenance metadata."""
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    section_id: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Generate a citation string for this chunk."""
        parts = [f"[Doc: {self.document_id}"]
        parts.append(f"p.{self.page_number}")
        if self.section_id:
            parts.append(f"sec: {self.section_id}")
        return ", ".join(parts) + "]"


@dataclass
class ExtractedTable:
    """A table extracted from a document."""
    table_id: str
    document_id: str
    page_number: int
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    section_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.headers and not self.rows:
            return ""
        lines = []
        if self.caption:
            lines.append(f"**{self.caption}**\n")
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        return "\n".join(lines)

    def to_dict_rows(self) -> List[Dict[str, str]]:
        """Convert to list of dictionaries."""
        if not self.headers:
            return [{"col_" + str(i): v for i, v in enumerate(row)} for row in self.rows]
        return [dict(zip(self.headers, row)) for row in self.rows]


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""
    document_id: str
    title: str
    author: Optional[str] = None
    date: Optional[datetime] = None
    page_count: int = 0
    word_count: int = 0
    document_type: DocumentType = DocumentType.PDF
    file_path: str = ""
    file_hash: str = ""
    language: str = "en"
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestedDocument:
    """Fully ingested and indexed document."""
    metadata: DocumentMetadata
    sections: List[DocumentSection]
    chunks: List[DocumentChunk]
    tables: List[ExtractedTable]
    pages: List[PageContent]


class DocumentIngestionPipeline:
    """
    Pipeline for ingesting documents with structure preservation.

    Pipeline stages:
    1. File parsing (PDF/DOCX/TXT)
    2. Structure analysis (sections, headings)
    3. Table extraction
    4. Chunk generation with overlap
    5. Embedding generation
    6. Index storage
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize ingestion pipeline.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between adjacent chunks
            embedding_model: Sentence transformer model for embeddings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self._embedder = None
        self._doc_counter = 0

        log.info(
            f"DocumentIngestionPipeline initialized "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model)
                log.info(f"Loaded embedding model: {self.embedding_model}")
            except ImportError:
                log.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedder

    def ingest_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestedDocument:
        """
        Ingest a single document through the full pipeline.

        Args:
            file_path: Path to the document file
            document_id: Optional custom ID (auto-generated if None)
            custom_metadata: Additional metadata to attach

        Returns:
            IngestedDocument with all extracted content
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Generate document ID
        if document_id is None:
            self._doc_counter += 1
            document_id = f"doc_{self._doc_counter:04d}"

        # Compute file hash for deduplication
        file_hash = self._compute_hash(file_path)

        # Determine document type
        doc_type = self._detect_type(path)

        log.info(f"Ingesting document: {path.name} (ID: {document_id})")

        # Stage 1: Parse file
        pages = self._parse_document(file_path, doc_type)

        # Stage 2: Extract structure
        sections = self._extract_sections(pages)

        # Stage 3: Extract tables
        tables = self._extract_tables(pages, document_id)

        # Stage 4: Generate chunks
        full_text = "\n".join(p.text for p in pages)
        chunks = self._generate_chunks(full_text, pages, sections, document_id)

        # Stage 5: Generate embeddings
        self._generate_embeddings(chunks)

        # Build metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            title=self._extract_title(pages, path),
            page_count=len(pages),
            word_count=len(full_text.split()),
            document_type=doc_type,
            file_path=str(path.absolute()),
            file_hash=file_hash,
            custom_metadata=custom_metadata or {},
        )

        log.info(
            f"Ingested {path.name}: {len(pages)} pages, "
            f"{len(sections)} sections, {len(tables)} tables, "
            f"{len(chunks)} chunks"
        )

        return IngestedDocument(
            metadata=metadata,
            sections=sections,
            chunks=chunks,
            tables=tables,
            pages=pages,
        )

    def ingest_collection(
        self,
        file_paths: List[str],
        collection_name: str = "default",
        fail_fast: bool = False,
    ) -> List[IngestedDocument]:
        """
        Ingest multiple documents into a collection.

        Args:
            file_paths: List of document file paths
            collection_name: Name for the document collection
            fail_fast: If True, raise on first failure. If False, collect
                       errors and report them at the end.

        Returns:
            List of IngestedDocument objects

        Raises:
            RuntimeError: If any documents failed to ingest (when fail_fast=False,
                          this is raised after processing all documents so partial
                          results are still available in the exception).
            Exception: Original exception if fail_fast=True
        """
        documents = []
        errors: List[Dict[str, str]] = []

        for i, path in enumerate(file_paths):
            doc_id = f"{collection_name}_{i+1:04d}"
            try:
                doc = self.ingest_document(path, document_id=doc_id)
                documents.append(doc)
            except Exception as e:
                log.error(f"Failed to ingest {path}: {e}")
                if fail_fast:
                    raise
                errors.append({"path": path, "error": str(e)})

        log.info(
            f"Collection '{collection_name}': ingested {len(documents)}/{len(file_paths)} documents"
        )

        if errors:
            error_summary = "; ".join(
                f"{e['path']}: {e['error']}" for e in errors
            )
            log.warning(
                f"Collection '{collection_name}' had {len(errors)} ingestion failures: "
                f"{error_summary}"
            )
            # Attach partial results to the exception so callers can access them
            exc = RuntimeError(
                f"{len(errors)}/{len(file_paths)} documents failed to ingest: {error_summary}"
            )
            exc.partial_results = documents  # type: ignore[attr-defined]
            exc.errors = errors  # type: ignore[attr-defined]
            raise exc

        return documents

    def _parse_document(
        self, file_path: str, doc_type: DocumentType
    ) -> List[PageContent]:
        """Parse document into pages based on type."""
        if doc_type == DocumentType.PDF:
            return self._parse_pdf(file_path)
        elif doc_type == DocumentType.TXT:
            return self._parse_text(file_path)
        elif doc_type == DocumentType.MD:
            return self._parse_text(file_path)
        elif doc_type == DocumentType.DOCX:
            return self._parse_docx(file_path)
        elif doc_type == DocumentType.CSV:
            return self._parse_csv(file_path)
        elif doc_type == DocumentType.XLSX:
            return self._parse_xlsx(file_path)
        elif doc_type == DocumentType.HTML:
            return self._parse_html(file_path)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")

    def _parse_pdf(self, file_path: str) -> List[PageContent]:
        """Parse PDF using PyMuPDF (fitz) for text and table extraction."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            log.error("PyMuPDF not installed. Install with: pip install pymupdf")
            raise

        doc = fitz.open(file_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")

            # Extract tables using PyMuPDF's table detection
            page_tables = []
            try:
                tabs = page.find_tables()
                for tab in tabs:
                    extracted = tab.extract()
                    if extracted and len(extracted) > 1:
                        headers = [str(h).strip() for h in extracted[0]]
                        rows = [
                            [str(cell).strip() for cell in row]
                            for row in extracted[1:]
                        ]
                        page_tables.append(
                            ExtractedTable(
                                table_id=f"table_p{page_num+1}_{len(page_tables)+1}",
                                document_id="",  # Set later
                                page_number=page_num + 1,
                                headers=headers,
                                rows=rows,
                            )
                        )
            except Exception as e:
                log.debug(f"Table extraction failed on page {page_num+1}: {e}")

            pages.append(
                PageContent(
                    page_number=page_num + 1,
                    text=text,
                    tables=page_tables,
                )
            )

        doc.close()
        return pages

    def _parse_text(self, file_path: str) -> List[PageContent]:
        """Parse plain text/markdown file as a single page."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [PageContent(page_number=1, text=text)]

    def _parse_docx(self, file_path: str) -> List[PageContent]:
        """Parse DOCX using python-docx."""
        try:
            from docx import Document
        except ImportError:
            log.error("python-docx not installed. Install with: pip install python-docx")
            raise

        doc = Document(file_path)
        text_parts = []
        for para in doc.paragraphs:
            text_parts.append(para.text)

        # DOCX doesn't have page boundaries, treat as single page
        return [PageContent(page_number=1, text="\n".join(text_parts))]

    def _parse_csv(self, file_path: str) -> List[PageContent]:
        """Parse CSV file into page content with table extraction."""
        import csv

        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return [PageContent(page_number=1, text="")]

        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        # Build text representation
        text_lines = [", ".join(headers)]
        for row in data_rows:
            text_lines.append(", ".join(row))

        # Create extracted table for structured access
        table = ExtractedTable(
            table_id="table_csv_1",
            document_id="",  # Set later
            page_number=1,
            headers=[str(h).strip() for h in headers],
            rows=[[str(cell).strip() for cell in row] for row in data_rows],
            caption=Path(file_path).stem,
        )

        return [PageContent(
            page_number=1,
            text="\n".join(text_lines),
            tables=[table],
        )]

    def _parse_xlsx(self, file_path: str) -> List[PageContent]:
        """Parse XLSX file with multiple sheets into page content."""
        try:
            import openpyxl
        except ImportError:
            log.error("openpyxl not installed. Install with: pip install openpyxl")
            raise

        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        pages = []

        for sheet_idx, sheet_name in enumerate(wb.sheetnames):
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                rows.append([str(cell) if cell is not None else "" for cell in row])

            if not rows:
                continue

            headers = rows[0]
            data_rows = rows[1:] if len(rows) > 1 else []

            text_lines = [f"Sheet: {sheet_name}"]
            text_lines.append(", ".join(headers))
            for row in data_rows:
                text_lines.append(", ".join(row))

            table = ExtractedTable(
                table_id=f"table_xlsx_{sheet_idx+1}",
                document_id="",  # Set later
                page_number=sheet_idx + 1,
                headers=[h.strip() for h in headers],
                rows=[[cell.strip() for cell in row] for row in data_rows],
                caption=sheet_name,
            )

            pages.append(PageContent(
                page_number=sheet_idx + 1,
                text="\n".join(text_lines),
                tables=[table],
            ))

        wb.close()
        return pages if pages else [PageContent(page_number=1, text="")]

    def _parse_html(self, file_path: str) -> List[PageContent]:
        """Parse HTML file, extracting text and tables."""
        try:
            from html.parser import HTMLParser
        except ImportError:
            pass  # Built-in module, always available

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()

        # Extract text using a simple HTML tag stripper
        import re
        # Remove script and style blocks
        clean = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", clean)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Extract tables using regex (lightweight, no BeautifulSoup dependency)
        tables = []
        table_pattern = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
        row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
        cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)

        for t_idx, table_match in enumerate(table_pattern.finditer(html_content)):
            table_html = table_match.group(1)
            rows = []
            for row_match in row_pattern.finditer(table_html):
                cells = [
                    re.sub(r"<[^>]+>", "", cell.group(1)).strip()
                    for cell in cell_pattern.finditer(row_match.group(1))
                ]
                if cells:
                    rows.append(cells)

            if rows:
                headers = rows[0]
                data_rows = rows[1:] if len(rows) > 1 else []
                tables.append(ExtractedTable(
                    table_id=f"table_html_{t_idx+1}",
                    document_id="",  # Set later
                    page_number=1,
                    headers=headers,
                    rows=data_rows,
                ))

        return [PageContent(page_number=1, text=text, tables=tables)]

    def _extract_sections(self, pages: List[PageContent]) -> List[DocumentSection]:
        """Extract document sections from heading patterns."""
        sections = []
        full_text = "\n".join(p.text for p in pages)

        # Detect headings by patterns (all caps, numbered sections, markdown)
        heading_patterns = [
            (r"^#{1}\s+(.+)$", 1),       # Markdown H1
            (r"^#{2}\s+(.+)$", 2),        # Markdown H2
            (r"^#{3}\s+(.+)$", 3),        # Markdown H3
            (r"^\d+\.\s+([A-Z].+)$", 1),  # Numbered sections
            (r"^\d+\.\d+\s+(.+)$", 2),    # Sub-sections
            (r"^([A-Z][A-Z\s]{5,})$", 1), # ALL CAPS headings
        ]

        lines = full_text.split("\n")
        char_offset = 0
        current_page = 1
        page_char_offsets = self._build_page_offsets(pages)

        for i, line in enumerate(lines):
            stripped = line.strip()
            for pattern, level in heading_patterns:
                match = re.match(pattern, stripped, re.MULTILINE)
                if match:
                    title = match.group(1).strip()
                    if len(title) > 3:  # Filter noise
                        page_num = self._offset_to_page(char_offset, page_char_offsets)
                        sections.append(
                            DocumentSection(
                                section_id=f"sec_{len(sections)+1:03d}",
                                title=title,
                                level=level,
                                content="",  # Filled in below
                                page_start=page_num,
                                page_end=page_num,
                                char_start=char_offset,
                                char_end=char_offset,
                            )
                        )
                    break
            char_offset += len(line) + 1  # +1 for newline

        # Fill in section content (text between this heading and the next)
        for i, section in enumerate(sections):
            if i + 1 < len(sections):
                section.content = full_text[section.char_start:sections[i+1].char_start]
                section.char_end = sections[i+1].char_start
                section.page_end = self._offset_to_page(
                    section.char_end, page_char_offsets
                )
            else:
                section.content = full_text[section.char_start:]
                section.char_end = len(full_text)
                section.page_end = len(pages)

        # Build hierarchy
        self._build_section_hierarchy(sections)

        return sections

    def _build_section_hierarchy(self, sections: List[DocumentSection]) -> None:
        """Link parent-child relationships in sections."""
        stack = []
        for section in sections:
            while stack and stack[-1].level >= section.level:
                stack.pop()
            if stack:
                section.parent_section_id = stack[-1].section_id
                stack[-1].children.append(section.section_id)
            stack.append(section)

    def _extract_tables(
        self, pages: List[PageContent], document_id: str
    ) -> List[ExtractedTable]:
        """Collect all tables from pages and assign document ID."""
        tables = []
        for page in pages:
            for table in page.tables:
                table.document_id = document_id
                tables.append(table)
        return tables

    def _generate_chunks(
        self,
        full_text: str,
        pages: List[PageContent],
        sections: List[DocumentSection],
        document_id: str,
    ) -> List[DocumentChunk]:
        """Generate overlapping text chunks with position metadata."""
        chunks = []
        page_offsets = self._build_page_offsets(pages)

        i = 0
        while i < len(full_text):
            end = min(i + self.chunk_size, len(full_text))

            # Try to break at sentence boundary
            if end < len(full_text):
                sentence_end = full_text.rfind(".", i, end)
                if sentence_end > i + self.chunk_size // 2:
                    end = sentence_end + 1

            chunk_text = full_text[i:end].strip()
            if not chunk_text:
                i = end
                continue

            page_num = self._offset_to_page(i, page_offsets)
            section_id = self._offset_to_section(i, sections)

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{len(chunks)+1:04d}",
                    document_id=document_id,
                    text=chunk_text,
                    page_number=page_num,
                    section_id=section_id,
                    char_start=i,
                    char_end=end,
                )
            )

            i = end - self.chunk_overlap

        return chunks

    # Class-level embedding cache: maps text hash -> embedding vector
    _embedding_cache: Dict[str, List[float]] = {}
    _EMBEDDING_CACHE_MAX_SIZE = 50_000

    def _generate_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings for all chunks, with caching.

        Cached embeddings are keyed by a hash of the chunk text to avoid
        redundant encoding of identical or previously-seen text.
        """
        embedder = self._get_embedder()
        if embedder is None:
            return

        uncached_indices = []
        uncached_texts = []

        for i, chunk in enumerate(chunks):
            cache_key = hashlib.md5(chunk.text.encode("utf-8")).hexdigest()
            cached = self._embedding_cache.get(cache_key)
            if cached is not None:
                chunk.embedding = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(chunk.text)

        if uncached_texts:
            log.debug(
                f"Embedding cache: {len(chunks) - len(uncached_texts)} hits, "
                f"{len(uncached_texts)} misses"
            )
            embeddings = embedder.encode(uncached_texts, show_progress_bar=False)

            for idx, emb in zip(uncached_indices, embeddings):
                emb_list = emb.tolist()
                chunks[idx].embedding = emb_list

                # Store in cache (evict oldest if needed)
                cache_key = hashlib.md5(chunks[idx].text.encode("utf-8")).hexdigest()
                if len(self._embedding_cache) >= self._EMBEDDING_CACHE_MAX_SIZE:
                    # Evict first entry (FIFO approximation)
                    first_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[first_key]
                self._embedding_cache[cache_key] = emb_list
        else:
            log.debug(f"All {len(chunks)} embeddings served from cache")

    def _build_page_offsets(self, pages: List[PageContent]) -> List[Tuple[int, int]]:
        """Build (start_offset, end_offset) for each page."""
        offsets = []
        current = 0
        for page in pages:
            end = current + len(page.text) + 1
            offsets.append((current, end))
            current = end
        return offsets

    def _offset_to_page(
        self, offset: int, page_offsets: List[Tuple[int, int]]
    ) -> int:
        """Convert character offset to page number."""
        for i, (start, end) in enumerate(page_offsets):
            if start <= offset < end:
                return i + 1
        return len(page_offsets)

    def _offset_to_section(
        self, offset: int, sections: List[DocumentSection]
    ) -> Optional[str]:
        """Find which section contains a character offset."""
        for section in reversed(sections):
            if section.char_start <= offset:
                return section.section_id
        return None

    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _detect_type(self, path: Path) -> DocumentType:
        """Detect document type from file extension."""
        ext_map = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".txt": DocumentType.TXT,
            ".md": DocumentType.MD,
            ".html": DocumentType.HTML,
            ".csv": DocumentType.CSV,
            ".xlsx": DocumentType.XLSX,
        }
        return ext_map.get(path.suffix.lower(), DocumentType.TXT)

    def _extract_title(self, pages: List[PageContent], path: Path) -> str:
        """Extract document title from first page or filename."""
        if pages and pages[0].text:
            first_lines = pages[0].text.strip().split("\n")
            for line in first_lines[:5]:
                stripped = line.strip()
                if stripped and len(stripped) > 5 and len(stripped) < 200:
                    return stripped
        return path.stem
```

---

## Document Store

### 1.2 Document Store with Vector Search

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Persistent document store with vector search capability.
"""

import json
import math
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from gaia.logger import get_logger

log = get_logger(__name__)


class DocumentStore:
    """
    Persistent multi-document store with vector similarity search.

    Stores document chunks with their embeddings in SQLite and provides
    cosine-similarity-based vector search for cross-document retrieval.

    Usage:
        store = DocumentStore("my_docs.db")

        # Store documents
        for doc in ingested_documents:
            store.store_document(doc)

        # Search across all documents
        results = store.search("revenue growth Q3", top_k=5)
        for chunk, score in results:
            print(f"[{chunk.document_id} p.{chunk.page_number}] {score:.3f}: {chunk.text[:100]}")
    """

    def __init__(self, db_path: str = "gaia_documents.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._embedder = None
        self._embedding_model = "all-MiniLM-L6-v2"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize document storage tables."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                title TEXT,
                author TEXT,
                page_count INTEGER,
                word_count INTEGER,
                document_type TEXT,
                file_path TEXT,
                file_hash TEXT UNIQUE,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                page_number INTEGER,
                section_id TEXT,
                char_start INTEGER,
                char_end INTEGER,
                embedding BLOB,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            );

            CREATE TABLE IF NOT EXISTS tables_extracted (
                table_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page_number INTEGER,
                headers TEXT,
                rows_data TEXT,
                caption TEXT,
                section_id TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_number);
            CREATE INDEX IF NOT EXISTS idx_tables_doc ON tables_extracted(document_id);
            CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(file_hash);
        """)
        conn.commit()
        conn.close()

    def _get_embedder(self):
        """Lazy-load sentence transformer for query embedding."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self._embedding_model)
            except ImportError:
                log.warning("sentence-transformers not installed for vector search")
        return self._embedder

    def store_document(self, doc: IngestedDocument) -> None:
        """Store an ingested document and its chunks."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Store document metadata
            cursor.execute("""
                INSERT OR REPLACE INTO documents
                (document_id, title, author, page_count, word_count,
                 document_type, file_path, file_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.metadata.document_id,
                doc.metadata.title,
                doc.metadata.author,
                doc.metadata.page_count,
                doc.metadata.word_count,
                doc.metadata.document_type.value,
                doc.metadata.file_path,
                doc.metadata.file_hash,
                json.dumps(doc.metadata.custom_metadata, default=str),
            ))

            # Store chunks with embeddings
            for chunk in doc.chunks:
                emb_blob = None
                if chunk.embedding:
                    import struct
                    emb_blob = struct.pack(f"{len(chunk.embedding)}f", *chunk.embedding)

                cursor.execute("""
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, document_id, text, page_number, section_id,
                     char_start, char_end, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.text,
                    chunk.page_number,
                    chunk.section_id,
                    chunk.char_start,
                    chunk.char_end,
                    emb_blob,
                    json.dumps(chunk.metadata, default=str),
                ))

            # Store tables
            for table in doc.tables:
                cursor.execute("""
                    INSERT OR REPLACE INTO tables_extracted
                    (table_id, document_id, page_number, headers, rows_data,
                     caption, section_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    table.table_id,
                    table.document_id,
                    table.page_number,
                    json.dumps(table.headers),
                    json.dumps(table.rows),
                    table.caption,
                    table.section_id,
                ))

            conn.commit()
            conn.close()

        log.info(
            f"Stored document {doc.metadata.document_id}: "
            f"{len(doc.chunks)} chunks, {len(doc.tables)} tables"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search across stored documents using vector similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            document_ids: Optional filter to specific documents

        Returns:
            List of (chunk, cosine_similarity_score) tuples, sorted by score desc
        """
        embedder = self._get_embedder()
        if embedder is None:
            log.warning("No embedder available; falling back to keyword search")
            return self._keyword_search(query, top_k, document_ids)

        # Encode query
        query_embedding = embedder.encode([query], show_progress_bar=False)[0].tolist()

        # Load all chunk embeddings and compute similarity
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        sql = "SELECT chunk_id, document_id, text, page_number, section_id, char_start, char_end, embedding FROM chunks WHERE embedding IS NOT NULL"
        params = []

        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            sql += f" AND document_id IN ({placeholders})"
            params.extend(document_ids)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        import struct
        for row in rows:
            emb_blob = row[7]
            if emb_blob is None:
                continue
            n_floats = len(emb_blob) // 4
            chunk_embedding = list(struct.unpack(f"{n_floats}f", emb_blob))

            score = self._cosine_similarity(query_embedding, chunk_embedding)

            chunk = DocumentChunk(
                chunk_id=row[0],
                document_id=row[1],
                text=row[2],
                page_number=row[3],
                section_id=row[4],
                char_start=row[5],
                char_end=row[6],
            )
            results.append((chunk, score))

        # Sort by similarity score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
    ) -> List[Tuple[DocumentChunk, float]]:
        """Fallback keyword search when embeddings are unavailable."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        sql = "SELECT chunk_id, document_id, text, page_number, section_id, char_start, char_end FROM chunks"
        params = []

        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            sql += f" WHERE document_id IN ({placeholders})"
            params.extend(document_ids)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        query_words = set(query.lower().split())
        results = []

        for row in rows:
            text = row[2]
            text_lower = text.lower()
            matches = sum(1 for w in query_words if w in text_lower)
            if matches > 0:
                score = matches / len(query_words)
                chunk = DocumentChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    text=text,
                    page_number=row[3],
                    section_id=row[4],
                    char_start=row[5],
                    char_end=row[6],
                )
                results.append((chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def has_document(self, file_hash: str) -> bool:
        """Check if a document with this hash is already stored (dedup)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM documents WHERE file_hash = ?", (file_hash,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def get_document_ids(self) -> List[str]:
        """Get all stored document IDs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT document_id FROM documents")
        ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return ids

    def delete_document(self, document_id: str) -> None:
        """Remove a document and all its chunks/tables."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM tables_extracted WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
            conn.close()
        log.info(f"Deleted document: {document_id}")
```

---

## Cross-Document Analysis Engine

### 2.1 Entity Resolution

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Cross-document entity resolution and linking.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from gaia.logger import get_logger

log = get_logger(__name__)


@dataclass
class Entity:
    """An entity extracted from documents."""
    entity_id: str
    name: str
    entity_type: str  # person, organization, date, metric, concept
    mentions: List["EntityMention"] = field(default_factory=list)
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def document_ids(self) -> Set[str]:
        return {m.document_id for m in self.mentions}

    @property
    def mention_count(self) -> int:
        return len(self.mentions)


@dataclass
class EntityMention:
    """A specific mention of an entity in a document."""
    document_id: str
    chunk_id: str
    page_number: int
    char_start: int
    char_end: int
    context: str  # Surrounding text
    confidence: float = 1.0


class EntityResolver:
    """
    Resolve and link entities across multiple documents.

    Identifies when different documents refer to the same entity
    (e.g., "AMD" and "Advanced Micro Devices").

    Supports two extraction backends:
    - Pattern-based (regex) - always available, no dependencies
    - spaCy NER - more accurate, requires 'pip install spacy'
      and a model download (e.g., 'python -m spacy download en_core_web_sm')
    """

    # Mapping from spaCy NER labels to our entity types
    SPACY_LABEL_MAP = {
        "PERSON": "person",
        "ORG": "organization",
        "GPE": "location",
        "LOC": "location",
        "DATE": "date",
        "TIME": "date",
        "MONEY": "metric",
        "PERCENT": "metric",
        "CARDINAL": "metric",
        "ORDINAL": "metric",
    }

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        self.similarity_threshold = similarity_threshold
        self._entities: Dict[str, Entity] = {}
        self._name_index: Dict[str, List[str]] = defaultdict(list)
        self._spacy_nlp = None
        self._use_spacy = use_spacy
        self._spacy_model = spacy_model

        if use_spacy:
            self._init_spacy()

    def _init_spacy(self) -> None:
        """Lazy-load spaCy NER model."""
        try:
            import spacy
            self._spacy_nlp = spacy.load(self._spacy_model)
            log.info(f"Loaded spaCy model: {self._spacy_model}")
        except ImportError:
            log.warning(
                "spaCy not installed. Falling back to regex-based NER. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
            self._spacy_nlp = None
        except OSError:
            log.warning(
                f"spaCy model '{self._spacy_model}' not found. Falling back to regex-based NER. "
                f"Download with: python -m spacy download {self._spacy_model}"
            )
            self._spacy_nlp = None

    def extract_entities(
        self,
        document: IngestedDocument,
        entity_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        """
        Extract entities from a single document.

        Uses spaCy NER if available, falling back to regex pattern matching.

        Args:
            document: Ingested document to analyze
            entity_types: Types to extract (default: all)

        Returns:
            List of extracted entities
        """
        entities = []
        allowed_types = set(entity_types) if entity_types else None

        for chunk in document.chunks:
            # Use spaCy NER if available
            if self._spacy_nlp is not None:
                spacy_entities = self._extract_with_spacy(
                    chunk, document.metadata.document_id, allowed_types
                )
                entities.extend(spacy_entities)

            # Always run pattern-based extraction for metric types
            # that spaCy may miss (fiscal periods, financial amounts, etc.)
            chunk_entities = self._extract_from_chunk(
                chunk, document.metadata.document_id, allowed_types
            )
            entities.extend(chunk_entities)

        # Deduplicate within document
        merged = self._merge_entities(entities)

        log.info(
            f"Extracted {len(merged)} entities from {document.metadata.document_id}"
        )
        return merged

    def _extract_with_spacy(
        self,
        chunk: DocumentChunk,
        document_id: str,
        allowed_types: Optional[Set[str]],
    ) -> List[Entity]:
        """Extract entities from a chunk using spaCy NER."""
        entities = []
        doc = self._spacy_nlp(chunk.text[:10000])  # Limit text length for performance

        for ent in doc.ents:
            entity_type = self.SPACY_LABEL_MAP.get(ent.label_)
            if entity_type is None:
                continue
            if allowed_types and entity_type not in allowed_types:
                continue

            mention = EntityMention(
                document_id=document_id,
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                char_start=chunk.char_start + ent.start_char,
                char_end=chunk.char_start + ent.end_char,
                context=chunk.text[max(0, ent.start_char - 50):ent.end_char + 50],
                confidence=0.9,  # spaCy entities get higher confidence
            )
            entities.append(
                Entity(
                    entity_id=f"ent_spacy_{len(entities)}",
                    name=ent.text.strip(),
                    entity_type=entity_type,
                    mentions=[mention],
                    attributes={"source": "spacy", "label": ent.label_},
                )
            )

        return entities

    def _extract_from_chunk(
        self,
        chunk: DocumentChunk,
        document_id: str,
        allowed_types: Optional[Set[str]],
    ) -> List[Entity]:
        """Extract entities from a single chunk using pattern matching."""
        import re

        entities = []
        text = chunk.text

        # Pattern-based extraction for common entity types
        patterns = {
            "metric": [
                # Financial metrics: $1.2B, $500M, 15%, etc.
                (r"\$[\d,.]+\s*[BMKbmk](?:illion)?", "financial_amount"),
                (r"\d+\.?\d*\s*%", "percentage"),
                (r"\d{4}\s*(?:Q[1-4]|FY)", "fiscal_period"),
            ],
            "date": [
                (r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", "full_date"),
                (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "short_date"),
                (r"\bQ[1-4]\s+\d{4}\b", "quarter"),
            ],
            "organization": [
                (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Inc|Corp|Ltd|LLC|Co)\b", "company"),
            ],
        }

        for entity_type, pattern_list in patterns.items():
            if allowed_types and entity_type not in allowed_types:
                continue

            for pattern, subtype in pattern_list:
                for match in re.finditer(pattern, text):
                    entity_name = match.group(0).strip()
                    mention = EntityMention(
                        document_id=document_id,
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        char_start=chunk.char_start + match.start(),
                        char_end=chunk.char_start + match.end(),
                        context=text[max(0, match.start()-50):match.end()+50],
                    )
                    entities.append(
                        Entity(
                            entity_id=f"ent_{len(entities)}",
                            name=entity_name,
                            entity_type=entity_type,
                            mentions=[mention],
                            attributes={"subtype": subtype},
                        )
                    )

        return entities

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge entities that refer to the same thing."""
        merged = {}
        for entity in entities:
            key = (entity.name.lower().strip(), entity.entity_type)
            if key in merged:
                merged[key].mentions.extend(entity.mentions)
                merged[key].aliases.add(entity.name)
            else:
                entity.entity_id = f"ent_{len(merged)+1:04d}"
                merged[key] = entity

        return list(merged.values())

    def resolve_across_documents(
        self,
        document_entities: Dict[str, List[Entity]],
    ) -> List[Entity]:
        """
        Resolve entities across multiple documents.

        Links entities from different documents that refer to the same thing.

        Args:
            document_entities: {document_id: [entities]} for each document

        Returns:
            List of resolved (merged) entities
        """
        all_entities = []
        for doc_id, entities in document_entities.items():
            all_entities.extend(entities)

        # Group by type, then fuzzy match names
        type_groups = defaultdict(list)
        for entity in all_entities:
            type_groups[entity.entity_type].append(entity)

        resolved = []
        for entity_type, group in type_groups.items():
            resolved.extend(self._fuzzy_merge(group))

        log.info(
            f"Resolved {len(all_entities)} entities into {len(resolved)} unique entities"
        )
        return resolved

    def _fuzzy_merge(self, entities: List[Entity]) -> List[Entity]:
        """Fuzzy merge entities based on name similarity.

        Optimization for large entity sets:
        - Entities are sorted by normalized name so that similar names
          are adjacent, allowing early termination of inner loop.
        - For very large sets (>500), a bucketing strategy groups entities
          by their first 3 characters, reducing comparisons from O(n^2)
          to O(n * k) where k is the average bucket size.
        """
        from difflib import SequenceMatcher

        if not entities:
            return []

        # For small sets, use the simple O(n^2) approach
        if len(entities) <= 500:
            return self._fuzzy_merge_simple(entities)

        # For large sets, bucket by normalized name prefix to reduce comparisons
        buckets: Dict[str, List[Tuple[int, Entity]]] = defaultdict(list)
        for i, entity in enumerate(entities):
            normalized = entity.name.lower().strip()
            # Use first 3 chars as bucket key; also add the entity to a
            # wildcard bucket for very short names
            key = normalized[:3] if len(normalized) >= 3 else normalized
            buckets[key].append((i, entity))

        merged = []
        used: Set[int] = set()

        for bucket_key, bucket_entities in buckets.items():
            # Compare within the bucket
            for bi, (i, e1) in enumerate(bucket_entities):
                if i in used:
                    continue

                group = [e1]
                used.add(i)
                name1 = e1.name.lower().strip()

                for bj, (j, e2) in enumerate(bucket_entities):
                    if j in used or j == i:
                        continue

                    similarity = SequenceMatcher(
                        None, name1, e2.name.lower().strip()
                    ).ratio()

                    if similarity >= self.similarity_threshold:
                        group.append(e2)
                        used.add(j)

                # Also check neighboring buckets for cross-bucket matches
                for other_key, other_bucket in buckets.items():
                    if other_key == bucket_key:
                        continue
                    # Quick prefix similarity check
                    if abs(len(other_key) - len(bucket_key)) > 2:
                        continue
                    prefix_sim = SequenceMatcher(None, bucket_key, other_key).ratio()
                    if prefix_sim < 0.5:
                        continue

                    for oj, (j, e2) in enumerate(other_bucket):
                        if j in used:
                            continue
                        similarity = SequenceMatcher(
                            None, name1, e2.name.lower().strip()
                        ).ratio()
                        if similarity >= self.similarity_threshold:
                            group.append(e2)
                            used.add(j)

                # Merge the group
                primary = group[0]
                for other in group[1:]:
                    primary.mentions.extend(other.mentions)
                    primary.aliases.add(other.name)
                    primary.aliases.update(other.aliases)

                merged.append(primary)

        return merged

    def _fuzzy_merge_simple(self, entities: List[Entity]) -> List[Entity]:
        """Simple O(n^2) fuzzy merge for small entity sets."""
        from difflib import SequenceMatcher

        merged = []
        used: Set[int] = set()

        for i, e1 in enumerate(entities):
            if i in used:
                continue

            group = [e1]
            used.add(i)

            for j, e2 in enumerate(entities):
                if j in used:
                    continue

                similarity = SequenceMatcher(
                    None,
                    e1.name.lower().strip(),
                    e2.name.lower().strip(),
                ).ratio()

                if similarity >= self.similarity_threshold:
                    group.append(e2)
                    used.add(j)

            # Merge the group into one entity
            primary = group[0]
            for other in group[1:]:
                primary.mentions.extend(other.mentions)
                primary.aliases.add(other.name)
                primary.aliases.update(other.aliases)

            merged.append(primary)

        return merged
```

### 2.2 Contradiction Detection

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Cross-document contradiction detection and consensus analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gaia.chat.sdk import ChatConfig, ChatSDK
from gaia.logger import get_logger

log = get_logger(__name__)


@dataclass
class Claim:
    """A factual claim extracted from a document."""
    claim_id: str
    text: str
    document_id: str
    chunk_id: str
    page_number: int
    section_id: Optional[str] = None
    confidence: float = 0.0
    entities: List[str] = field(default_factory=list)
    claim_type: str = "factual"  # factual, opinion, prediction


@dataclass
class ContradictionPair:
    """A pair of contradicting claims."""
    contradiction_id: str
    claim_a: Claim
    claim_b: Claim
    contradiction_type: str  # "direct", "numeric", "temporal", "logical"
    severity: str  # "critical", "major", "minor"
    explanation: str
    confidence: float = 0.0

    @property
    def summary(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.contradiction_type} contradiction:\n"
            f"  Doc {self.claim_a.document_id} (p.{self.claim_a.page_number}): "
            f"{self.claim_a.text[:100]}\n"
            f"  Doc {self.claim_b.document_id} (p.{self.claim_b.page_number}): "
            f"{self.claim_b.text[:100]}\n"
            f"  Explanation: {self.explanation}"
        )


@dataclass
class ConsensusResult:
    """Result of consensus analysis across documents."""
    topic: str
    consensus_claim: str
    supporting_documents: List[str]
    contradicting_documents: List[str]
    confidence: float
    evidence: List[Claim] = field(default_factory=list)


class ContradictionDetector:
    """
    Detect contradictions and build consensus across documents.

    Pipeline:
    1. Extract claims from each document
    2. Cluster related claims by topic/entity
    3. Compare claim pairs using LLM
    4. Score contradiction severity
    5. Build consensus view
    """

    def __init__(
        self,
        model: str = "Qwen3-Coder-30B-A3B-Instruct-GGUF",
        use_claude: bool = False,
    ):
        config = ChatConfig(
            model=model,
            max_tokens=1024,
            use_claude=use_claude,
            system_prompt=self._system_prompt(),
        )
        self.chat = ChatSDK(config)
        self._claim_counter = 0
        self._contradiction_counter = 0

    def _system_prompt(self) -> str:
        return """You are an expert fact-checker and analyst. Your job is to:
1. Extract factual claims from text
2. Compare claims from different sources
3. Detect contradictions, inconsistencies, and agreements
4. Provide confidence scores for your assessments

Always respond in valid JSON format."""

    async def extract_claims(
        self,
        document: IngestedDocument,
        topic_filter: Optional[str] = None,
    ) -> List[Claim]:
        """
        Extract factual claims from a document.

        Args:
            document: Ingested document
            topic_filter: Only extract claims related to this topic

        Returns:
            List of extracted claims
        """
        claims = []

        for chunk in document.chunks:
            prompt = f"""Extract key factual claims from this text.
{f'Focus on claims related to: {topic_filter}' if topic_filter else ''}

Text:
{chunk.text[:2000]}

Return a JSON array of claims, each with:
- "text": the claim as a clear statement
- "type": "factual" | "opinion" | "prediction"
- "confidence": 0.0-1.0
- "entities": list of key entities mentioned"""

            try:
                import json
                response = self.chat.send(prompt)
                claim_data = json.loads(response.text)

                if isinstance(claim_data, list):
                    for item in claim_data:
                        self._claim_counter += 1
                        claims.append(
                            Claim(
                                claim_id=f"claim_{self._claim_counter:04d}",
                                text=item.get("text", ""),
                                document_id=document.metadata.document_id,
                                chunk_id=chunk.chunk_id,
                                page_number=chunk.page_number,
                                section_id=chunk.section_id,
                                confidence=item.get("confidence", 0.5),
                                entities=item.get("entities", []),
                                claim_type=item.get("type", "factual"),
                            )
                        )
            except Exception as e:
                log.debug(f"Claim extraction failed for chunk {chunk.chunk_id}: {e}")

        log.info(
            f"Extracted {len(claims)} claims from {document.metadata.document_id}"
        )
        return claims

    async def detect_contradictions(
        self,
        claims_by_document: Dict[str, List[Claim]],
    ) -> List[ContradictionPair]:
        """
        Detect contradictions between claims from different documents.

        Args:
            claims_by_document: {document_id: [claims]}

        Returns:
            List of detected contradiction pairs
        """
        contradictions = []

        # Get all document pairs
        doc_ids = list(claims_by_document.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc_a_claims = claims_by_document[doc_ids[i]]
                doc_b_claims = claims_by_document[doc_ids[j]]

                # Compare relevant claim pairs
                pairs = self._find_related_pairs(doc_a_claims, doc_b_claims)

                for claim_a, claim_b in pairs:
                    result = await self._compare_claims(claim_a, claim_b)
                    if result:
                        contradictions.append(result)

        log.info(f"Detected {len(contradictions)} contradictions")
        return contradictions

    def _find_related_pairs(
        self, claims_a: List[Claim], claims_b: List[Claim]
    ) -> List[Tuple[Claim, Claim]]:
        """Find pairs of claims that discuss related topics."""
        pairs = []

        for ca in claims_a:
            for cb in claims_b:
                # Check entity overlap
                entities_a = set(e.lower() for e in ca.entities)
                entities_b = set(e.lower() for e in cb.entities)
                overlap = entities_a & entities_b

                if overlap:
                    pairs.append((ca, cb))
                    continue

                # Check keyword overlap
                words_a = set(ca.text.lower().split())
                words_b = set(cb.text.lower().split())
                common = words_a & words_b
                # Filter common English words
                common -= {"the", "a", "an", "is", "was", "are", "in", "of", "to", "and", "for", "that", "this"}
                if len(common) >= 3:
                    pairs.append((ca, cb))

        return pairs

    async def _compare_claims(
        self, claim_a: Claim, claim_b: Claim
    ) -> Optional[ContradictionPair]:
        """Compare two claims for contradiction using LLM."""
        import json

        prompt = f"""Compare these two claims from different documents:

Claim A (from {claim_a.document_id}, page {claim_a.page_number}):
"{claim_a.text}"

Claim B (from {claim_b.document_id}, page {claim_b.page_number}):
"{claim_b.text}"

Do these claims contradict each other? Respond with JSON:
{{
  "contradicts": true/false,
  "type": "direct" | "numeric" | "temporal" | "logical" | "none",
  "severity": "critical" | "major" | "minor",
  "explanation": "brief explanation",
  "confidence": 0.0-1.0
}}"""

        try:
            response = self.chat.send(prompt)
            result = json.loads(response.text)

            if result.get("contradicts") and result.get("confidence", 0) > 0.5:
                self._contradiction_counter += 1
                return ContradictionPair(
                    contradiction_id=f"contra_{self._contradiction_counter:04d}",
                    claim_a=claim_a,
                    claim_b=claim_b,
                    contradiction_type=result.get("type", "direct"),
                    severity=result.get("severity", "minor"),
                    explanation=result.get("explanation", ""),
                    confidence=result.get("confidence", 0.5),
                )
        except Exception as e:
            log.debug(f"Claim comparison failed: {e}")

        return None

    async def build_consensus(
        self,
        claims_by_document: Dict[str, List[Claim]],
        topic: str,
    ) -> ConsensusResult:
        """
        Build a consensus view on a topic across all documents.

        Args:
            claims_by_document: All claims organized by document
            topic: Topic to analyze

        Returns:
            ConsensusResult with agreement/disagreement analysis
        """
        import json

        # Gather all relevant claims
        relevant_claims = []
        for doc_id, claims in claims_by_document.items():
            for claim in claims:
                if topic.lower() in claim.text.lower():
                    relevant_claims.append(claim)

        if not relevant_claims:
            return ConsensusResult(
                topic=topic,
                consensus_claim="No relevant claims found",
                supporting_documents=[],
                contradicting_documents=[],
                confidence=0.0,
            )

        claims_text = "\n".join(
            f"- [{c.document_id}, p.{c.page_number}]: {c.text}"
            for c in relevant_claims
        )

        prompt = f"""Analyze these claims about "{topic}" from different documents:

{claims_text}

Provide a consensus analysis as JSON:
{{
  "consensus_claim": "The overall consensus statement",
  "supporting_documents": ["doc_ids that agree"],
  "contradicting_documents": ["doc_ids that disagree"],
  "confidence": 0.0-1.0,
  "reasoning": "explanation"
}}"""

        try:
            response = self.chat.send(prompt)
            result = json.loads(response.text)

            return ConsensusResult(
                topic=topic,
                consensus_claim=result.get("consensus_claim", ""),
                supporting_documents=result.get("supporting_documents", []),
                contradicting_documents=result.get("contradicting_documents", []),
                confidence=result.get("confidence", 0.5),
                evidence=relevant_claims,
            )
        except Exception as e:
            log.error(f"Consensus building failed: {e}")
            return ConsensusResult(
                topic=topic,
                consensus_claim="Analysis failed",
                supporting_documents=[],
                contradicting_documents=[],
                confidence=0.0,
            )
```

---

## Citation Tracking System

### 3.1 Citation Tracker

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Citation tracking system for evidence-backed responses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CitationFormat(Enum):
    """Supported citation output formats."""
    INLINE = "inline"       # [Doc A, p.5]
    FOOTNOTE = "footnote"   # [1] with footnotes
    APA = "apa"             # APA style
    IEEE = "ieee"           # IEEE style
    MARKDOWN = "markdown"   # Markdown links


@dataclass
class Citation:
    """A single citation reference."""
    citation_id: str
    document_id: str
    document_title: str
    page_number: int
    section_title: Optional[str] = None
    paragraph_index: Optional[int] = None
    quote: Optional[str] = None  # Direct quote from source
    relevance_score: float = 0.0

    def format(self, style: CitationFormat = CitationFormat.INLINE) -> str:
        """Format citation in specified style."""
        if style == CitationFormat.INLINE:
            parts = [self.document_title, f"p.{self.page_number}"]
            if self.section_title:
                parts.append(f'"{self.section_title}"')
            return f"[{', '.join(parts)}]"

        elif style == CitationFormat.FOOTNOTE:
            return f"[{self.citation_id}]"

        elif style == CitationFormat.APA:
            return f"({self.document_title}, p. {self.page_number})"

        elif style == CitationFormat.MARKDOWN:
            return f"[{self.document_title}](#{self.document_id})"

        return f"[{self.document_id}:{self.page_number}]"


@dataclass
class CitedPassage:
    """A passage of text with its supporting citations."""
    text: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0

    def format_with_citations(
        self, style: CitationFormat = CitationFormat.INLINE
    ) -> str:
        """Return text with inline citations."""
        citation_refs = " ".join(c.format(style) for c in self.citations)
        return f"{self.text} {citation_refs}"


@dataclass
class SynthesisReport:
    """A complete synthesis report with citations."""
    title: str
    summary: str
    passages: List[CitedPassage]
    all_citations: List[Citation]
    contradictions: List[ContradictionPair] = field(default_factory=list)
    consensus_results: List[ConsensusResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(
        self, citation_style: CitationFormat = CitationFormat.INLINE
    ) -> str:
        """Render report as markdown with citations."""
        lines = [f"# {self.title}\n"]
        lines.append(f"## Summary\n\n{self.summary}\n")

        lines.append("## Analysis\n")
        for passage in self.passages:
            lines.append(passage.format_with_citations(citation_style))
            lines.append("")

        if self.contradictions:
            lines.append("## Contradictions Found\n")
            for c in self.contradictions:
                lines.append(f"- **{c.severity.upper()}**: {c.explanation}")
                lines.append(f"  - Source A: {c.claim_a.document_id}, p.{c.claim_a.page_number}")
                lines.append(f"  - Source B: {c.claim_b.document_id}, p.{c.claim_b.page_number}")
                lines.append("")

        if self.consensus_results:
            lines.append("## Consensus Views\n")
            for cr in self.consensus_results:
                lines.append(f"### {cr.topic}")
                lines.append(f"{cr.consensus_claim}")
                lines.append(f"Confidence: {cr.confidence:.0%}")
                lines.append(f"Supporting: {', '.join(cr.supporting_documents)}")
                if cr.contradicting_documents:
                    lines.append(f"Contradicting: {', '.join(cr.contradicting_documents)}")
                lines.append("")

        # References section
        lines.append("## References\n")
        seen = set()
        for citation in self.all_citations:
            key = (citation.document_id, citation.page_number)
            if key not in seen:
                seen.add(key)
                lines.append(
                    f"- [{citation.citation_id}] {citation.document_title}, "
                    f"Page {citation.page_number}"
                )
                if citation.section_title:
                    lines.append(f"  Section: {citation.section_title}")

        return "\n".join(lines)

    def get_bibliography(self) -> List[Dict[str, str]]:
        """Get a deduplicated bibliography."""
        seen = set()
        bibliography = []
        for citation in self.all_citations:
            if citation.document_id not in seen:
                seen.add(citation.document_id)
                bibliography.append({
                    "id": citation.document_id,
                    "title": citation.document_title,
                    "pages_cited": sorted(set(
                        c.page_number for c in self.all_citations
                        if c.document_id == citation.document_id
                    )),
                })
        return bibliography
```

---

## Integration with GAIA

### 4.1 SynthesisAgent

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
SynthesisAgent - GAIA agent for multi-document analysis and synthesis.
"""

import json
from typing import Any, Dict, List, Optional

from gaia.agents.base import Agent
from gaia.agents.base.tools import tool
from gaia.logger import get_logger

log = get_logger(__name__)


class SynthesisAgent(Agent):
    """
    GAIA agent for multi-document synthesis and analysis.

    Capabilities:
    - Ingest document collections
    - Cross-document search with citations
    - Contradiction detection
    - Table extraction and comparison
    - Evidence synthesis reports
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs,
    ):
        self.pipeline = DocumentIngestionPipeline(
            chunk_size=chunk_size,
            embedding_model=embedding_model,
        )
        self.entity_resolver = EntityResolver()
        self.contradiction_detector = ContradictionDetector()

        # Document storage
        self._documents: Dict[str, IngestedDocument] = {}
        self._collections: Dict[str, List[str]] = {}

        super().__init__(**kwargs)

    def _get_system_prompt(self) -> str:
        return """You are a multi-document analysis expert. You can:
1. Ingest and index document collections
2. Search across all documents with citation tracking
3. Detect contradictions between sources
4. Extract and compare tables
5. Generate synthesis reports with evidence

Always cite your sources with document name and page number.
When comparing documents, highlight agreements AND disagreements."""

    def _register_tools(self):
        """Register synthesis-specific tools."""

        @tool
        def ingest_documents(
            file_paths: str,
            collection_name: str = "default",
        ) -> Dict[str, Any]:
            """
            Ingest a collection of documents for analysis.

            Args:
                file_paths: Comma-separated file paths to ingest
                collection_name: Name for this document collection

            Returns:
                Ingestion summary with document count and stats
            """
            paths = [p.strip() for p in file_paths.split(",")]
            documents = self.pipeline.ingest_collection(paths, collection_name)

            for doc in documents:
                self._documents[doc.metadata.document_id] = doc

            self._collections[collection_name] = [
                d.metadata.document_id for d in documents
            ]

            return {
                "status": "success",
                "collection": collection_name,
                "documents_ingested": len(documents),
                "total_pages": sum(d.metadata.page_count for d in documents),
                "total_tables": sum(len(d.tables) for d in documents),
                "total_chunks": sum(len(d.chunks) for d in documents),
                "documents": [
                    {
                        "id": d.metadata.document_id,
                        "title": d.metadata.title,
                        "pages": d.metadata.page_count,
                        "tables": len(d.tables),
                    }
                    for d in documents
                ],
            }

        @tool
        def search_documents(
            query: str,
            collection_name: str = "default",
            max_results: int = 5,
        ) -> Dict[str, Any]:
            """
            Search across all documents in a collection with citations.

            Args:
                query: Search query
                collection_name: Collection to search in
                max_results: Maximum results to return

            Returns:
                Search results with citations and page references
            """
            doc_ids = self._collections.get(collection_name, [])
            if not doc_ids:
                return {"status": "error", "message": f"Collection not found: {collection_name}"}

            results = []
            for doc_id in doc_ids:
                doc = self._documents.get(doc_id)
                if not doc:
                    continue

                for chunk in doc.chunks:
                    if self._text_matches(query, chunk.text):
                        results.append({
                            "document_id": doc_id,
                            "document_title": doc.metadata.title,
                            "page": chunk.page_number,
                            "section": chunk.section_id,
                            "text": chunk.text[:500],
                            "citation": chunk.citation,
                        })

            # Sort by relevance and limit
            results = results[:max_results]

            return {
                "status": "success",
                "query": query,
                "results": results,
                "total_found": len(results),
            }

        @tool
        def compare_tables(
            collection_name: str = "default",
            metric_name: str = "",
        ) -> Dict[str, Any]:
            """
            Extract and compare tables across documents in a collection.

            Args:
                collection_name: Collection to analyze
                metric_name: Specific metric to compare (empty for all)

            Returns:
                Table comparison results with source citations
            """
            doc_ids = self._collections.get(collection_name, [])
            all_tables = []

            for doc_id in doc_ids:
                doc = self._documents.get(doc_id)
                if doc:
                    for table in doc.tables:
                        all_tables.append({
                            "document_id": doc_id,
                            "document_title": doc.metadata.title,
                            "page": table.page_number,
                            "headers": table.headers,
                            "rows": table.rows[:10],
                            "caption": table.caption,
                            "markdown": table.to_markdown(),
                        })

            return {
                "status": "success",
                "collection": collection_name,
                "table_count": len(all_tables),
                "tables": all_tables,
            }

        @tool
        def detect_contradictions(
            collection_name: str = "default",
            topic: str = "",
        ) -> Dict[str, Any]:
            """
            Detect contradictions between documents in a collection.

            Args:
                collection_name: Collection to analyze
                topic: Optional topic filter

            Returns:
                List of contradictions with severity and citations
            """
            import asyncio

            doc_ids = self._collections.get(collection_name, [])
            claims_by_doc = {}

            loop = asyncio.new_event_loop()
            try:
                for doc_id in doc_ids:
                    doc = self._documents.get(doc_id)
                    if doc:
                        claims = loop.run_until_complete(
                            self.contradiction_detector.extract_claims(
                                doc, topic_filter=topic if topic else None
                            )
                        )
                        claims_by_doc[doc_id] = claims

                contradictions = loop.run_until_complete(
                    self.contradiction_detector.detect_contradictions(claims_by_doc)
                )
            finally:
                loop.close()

            return {
                "status": "success",
                "contradiction_count": len(contradictions),
                "contradictions": [
                    {
                        "severity": c.severity,
                        "type": c.contradiction_type,
                        "doc_a": c.claim_a.document_id,
                        "page_a": c.claim_a.page_number,
                        "claim_a": c.claim_a.text,
                        "doc_b": c.claim_b.document_id,
                        "page_b": c.claim_b.page_number,
                        "claim_b": c.claim_b.text,
                        "explanation": c.explanation,
                        "confidence": c.confidence,
                    }
                    for c in contradictions
                ],
            }

        @tool
        def generate_synthesis_report(
            collection_name: str = "default",
            query: str = "",
            include_contradictions: bool = True,
        ) -> Dict[str, Any]:
            """
            Generate a comprehensive synthesis report from multiple documents.

            Args:
                collection_name: Collection to synthesize
                query: Focus topic/question for the report
                include_contradictions: Include contradiction analysis

            Returns:
                Synthesis report with citations in markdown
            """
            doc_ids = self._collections.get(collection_name, [])
            if not doc_ids:
                return {"status": "error", "message": "Collection not found"}

            # Gather evidence
            all_relevant_chunks = []
            for doc_id in doc_ids:
                doc = self._documents.get(doc_id)
                if doc:
                    for chunk in doc.chunks:
                        if not query or self._text_matches(query, chunk.text):
                            all_relevant_chunks.append(chunk)

            if not all_relevant_chunks:
                return {"status": "error", "message": "No relevant content found"}

            # Build report using LLM
            evidence_text = "\n\n".join(
                f"[{c.document_id}, p.{c.page_number}]: {c.text[:500]}"
                for c in all_relevant_chunks[:20]
            )

            prompt = (
                f"Based on these sources, write a synthesis report about: {query}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"Include citations in [Document, p.X] format."
            )

            from gaia.chat.sdk import ChatConfig, ChatSDK

            config = ChatConfig(max_tokens=2048)
            chat = ChatSDK(config)
            response = chat.send(prompt)

            report = {
                "status": "success",
                "title": f"Synthesis Report: {query}",
                "report": response.text,
                "sources_used": len(all_relevant_chunks),
                "documents_analyzed": len(doc_ids),
            }

            return report

    def _text_matches(self, query: str, text: str) -> bool:
        """Simple keyword matching (replaced by vector search in production)."""
        query_words = set(query.lower().split())
        text_lower = text.lower()
        matches = sum(1 for w in query_words if w in text_lower)
        return matches >= max(1, len(query_words) // 2)
```

---

## Safety and Security Considerations

1. **Document Access Control**: Only process documents the user has explicit access to
2. **Content Sanitization**: Strip executable content from ingested documents
3. **Memory Limits**: Cap document collection size to prevent OOM (default: 100 docs, 1GB)
4. **Citation Integrity**: Never fabricate citations; verify every reference maps to real content
5. **PII Detection**: Scan for and flag personally identifiable information in reports
6. **Local Processing**: All analysis runs on local LLM; document content never leaves the machine

---

## Implementation Plan

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|-------------|
| 1-2 | Ingestion | PDF parser, structure analyzer, table extractor | DocumentIngestionPipeline |
| 3-4 | Storage | ChunkIndex, embedding generation, vector search | Multi-document search |
| 5-6 | Analysis | Entity resolution, claim extraction, contradiction detection | EntityResolver, ContradictionDetector |
| 7 | Citations | Citation tracker, reference formatter, evidence chains | CitationTracker |
| 8 | Synthesis | Report generator, consensus builder, table comparison | SynthesisReport |
| 9 | Agent | SynthesisAgent, tool registration, CLI integration | Full agent |
| 10 | Polish | Testing, optimization, documentation | Production-ready |

---

## Testing Strategy

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for multi-document synthesis."""

import pytest
import tempfile
import os


class TestDocumentIngestion:
    """Test document ingestion pipeline."""

    @pytest.fixture
    def pipeline(self):
        return DocumentIngestionPipeline(chunk_size=500, chunk_overlap=100)

    @pytest.fixture
    def sample_text_file(self, tmp_path):
        content = """# Introduction

This document discusses revenue growth in Q3 2025.

## Financial Results

Revenue was $1.5 billion, up 15% year-over-year.
Operating income reached $400 million.

## Outlook

We project Q4 revenue of $1.7 billion.
"""
        file_path = tmp_path / "test_doc.txt"
        file_path.write_text(content)
        return str(file_path)

    def test_ingest_text_file(self, pipeline, sample_text_file):
        doc = pipeline.ingest_document(sample_text_file)
        assert doc.metadata.page_count == 1
        assert len(doc.chunks) > 0
        assert doc.metadata.word_count > 0

    def test_section_extraction(self, pipeline, sample_text_file):
        doc = pipeline.ingest_document(sample_text_file)
        assert len(doc.sections) >= 2
        assert any("Introduction" in s.title for s in doc.sections)

    def test_chunk_has_page_reference(self, pipeline, sample_text_file):
        doc = pipeline.ingest_document(sample_text_file)
        for chunk in doc.chunks:
            assert chunk.page_number >= 1
            assert chunk.document_id is not None

    def test_collection_ingestion(self, pipeline, tmp_path):
        for i in range(3):
            (tmp_path / f"doc_{i}.txt").write_text(f"Document {i} content about topic X.")
        paths = [str(tmp_path / f"doc_{i}.txt") for i in range(3)]
        docs = pipeline.ingest_collection(paths, "test_collection")
        assert len(docs) == 3


class TestTableExtraction:
    """Test table extraction and comparison."""

    def test_table_to_markdown(self):
        table = ExtractedTable(
            table_id="t1",
            document_id="doc1",
            page_number=1,
            headers=["Metric", "Q3", "Q4"],
            rows=[["Revenue", "$1.5B", "$1.7B"], ["Profit", "$400M", "$500M"]],
        )
        md = table.to_markdown()
        assert "Revenue" in md
        assert "$1.5B" in md
        assert "|" in md

    def test_table_to_dict_rows(self):
        table = ExtractedTable(
            table_id="t1",
            document_id="doc1",
            page_number=1,
            headers=["Name", "Value"],
            rows=[["A", "1"], ["B", "2"]],
        )
        rows = table.to_dict_rows()
        assert rows[0]["Name"] == "A"
        assert rows[1]["Value"] == "2"


class TestEntityResolution:
    """Test cross-document entity resolution."""

    def test_merge_same_entity(self):
        resolver = EntityResolver(similarity_threshold=0.85)
        entities = [
            Entity(entity_id="e1", name="AMD", entity_type="organization"),
            Entity(entity_id="e2", name="AMD", entity_type="organization"),
        ]
        merged = resolver._merge_entities(entities)
        assert len(merged) == 1
        assert merged[0].mention_count == 0  # No mentions added in test
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Ingestion speed | >10 pages/second for PDF | Benchmark on sample documents |
| Citation accuracy | >95% citations map to real content | Manual verification on 100 citations |
| Contradiction detection precision | >80% | Expert review of detected pairs |
| Table extraction accuracy | >90% cell accuracy | Comparison with manual extraction |
| Cross-doc search relevance | >0.7 NDCG@10 | Relevance judgments |
| Entity resolution F1 | >0.85 | Gold-standard entity labels |
| Report quality rating | >7/10 human rating | Expert evaluation |

---

## Complete Code

```
src/gaia/synthesis/
    __init__.py
    pipeline.py          # DocumentIngestionPipeline
    models.py            # All data models
    store.py             # DocumentStore, ChunkIndex
    entity.py            # EntityResolver
    contradiction.py     # ContradictionDetector
    citation.py          # CitationTracker, Citation, SynthesisReport
    table.py             # TableExtractor, table comparison
    agent.py             # SynthesisAgent
    report.py            # ReportGenerator

tests/unit/synthesis/
    test_pipeline.py
    test_entity.py
    test_contradiction.py
    test_citation.py
    test_table.py

tests/integration/synthesis/
    test_synthesis_agent.py
    test_pdf_collection.py
```
