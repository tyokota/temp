#!/usr/bin/env python3
“””
Local Email Archive System with Semantic Search
Archives emails from Outlook to SQLite with FAISS vector search capabilities.
“””

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from contextlib import contextmanager

import numpy as np
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============ CONFIGURATION ============

@dataclass
class Config:
“”“System configuration with validation.”””

```
# Outlook settings
outlook_store_hint: str = ""
outlook_folder_path: str = "Inbox"

# Storage paths
db_path: str = "mailbox.db"
faiss_path: str = "mailbox.faiss"
faiss_map_path: str = "faiss_map.json"
cache_dir: str = ".emb_cache"

# Model settings
embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

# Processing limits
max_to_ingest: int = 5000
batch_embed: int = 256
preview_limit: int = 260

def __post_init__(self):
    """Validate configuration and create necessary directories."""
    self._ensure_paths()
    self._validate_config()

def _ensure_paths(self):
    """Create necessary directories."""
    for path_attr in ['db_path', 'faiss_path', 'faiss_map_path']:
        path = getattr(self, path_attr)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    Path(self.cache_dir).mkdir(exist_ok=True)

def _validate_config(self):
    """Validate configuration values."""
    if self.batch_embed <= 0:
        raise ValueError("batch_embed must be positive")
    if self.max_to_ingest <= 0:
        raise ValueError("max_to_ingest must be positive")
```

# ============ LOGGING SETUP ============

def setup_logging(level: str = “INFO”) -> logging.Logger:
“”“Configure logging with appropriate format.”””
logging.basicConfig(
level=getattr(logging, level.upper()),
format=’%(asctime)s - %(name)s - %(levelname)s - %(message)s’,
datefmt=’%H:%M:%S’
)
return logging.getLogger(**name**)

# ============ UTILITIES ============

def compute_hash(text: str) -> str:
“”“Compute SHA1 hash of text for stable IDs.”””
return hashlib.sha1(text.encode(“utf-8”, errors=“ignore”)).hexdigest()

def clean_text(text: Optional[str]) -> str:
“”“Clean and normalize text content.”””
if not text:
return “”
return re.sub(r’\s+’, ’ ‘, text.strip().replace(’\r\n’, ‘\n’))

def preview_text(text: str, limit: int = 260) -> str:
“”“Create a preview of text with length limit.”””
cleaned = clean_text(text)
return f”{cleaned[:limit]}…” if len(cleaned) > limit else cleaned

# ============ TEXT PROCESSING ============

class TextProcessor:
“”“Handles HTML to text conversion and cleaning.”””

```
def __init__(self):
    try:
        import html2text
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = True
        self.html2text.ignore_images = True
        self.html2text.body_width = 0
        self.has_html2text = True
    except ImportError:
        self.has_html2text = False
        logging.warning("html2text not available, using BeautifulSoup only")

def html_to_text(self, html_content: str) -> str:
    """Convert HTML to plain text."""
    if not html_content:
        return ""
    
    # Check if content is actually HTML
    html_indicators = ['<html', '</p>', '<div', '<br', '<table']
    is_html = any(indicator in html_content.lower() for indicator in html_indicators)
    
    if not is_html:
        return html_content
    
    try:
        if self.has_html2text:
            return self.html2text.handle(html_content)
        else:
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text("\n")
    except Exception as e:
        logging.warning(f"HTML parsing failed: {e}")
        return html_content
```

# ============ DATABASE OPERATIONS ============

class EmailDatabase:
“”“Handles all database operations for email storage.”””

```
DDL = """
CREATE TABLE IF NOT EXISTS emails(
    id TEXT PRIMARY KEY,
    message_id TEXT,
    subject TEXT,
    from_addr TEXT,
    to_addrs TEXT,
    cc_addrs TEXT,
    sent_at TEXT,
    folder TEXT,
    headers TEXT,
    body TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_emails_sent_at ON emails(sent_at);
CREATE INDEX IF NOT EXISTS idx_emails_from_addr ON emails(from_addr);
CREATE INDEX IF NOT EXISTS idx_emails_folder ON emails(folder);
"""

def __init__(self, db_path: str, logger: logging.Logger):
    self.db_path = db_path
    self.logger = logger
    self._init_database()

def _init_database(self):
    """Initialize database with schema."""
    with self.get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        for stmt in self.DDL.strip().split(';'):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)

@contextmanager
def get_connection(self):
    """Context manager for database connections."""
    conn = sqlite3.connect(self.db_path)
    try:
        yield conn
    finally:
        conn.close()

def upsert_emails(self, emails: List[Dict]) -> int:
    """Insert or update multiple emails efficiently."""
    if not emails:
        return 0
    
    with self.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.executemany("""
                INSERT OR IGNORE INTO emails
                (id, message_id, subject, from_addr, to_addrs, cc_addrs, 
                 sent_at, folder, headers, body)
                VALUES (:id, :message_id, :subject, :from_addr, :to_addrs, 
                        :cc_addrs, :sent_at, :folder, :headers, :body)
            """, emails)
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            self.logger.error(f"Database upsert failed: {e}")
            conn.rollback()
            raise

def get_emails_needing_vectors(self, cache_dir: str) -> List[Tuple[str, str]]:
    """Get emails that need vector embeddings."""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, subject, body FROM emails")
        
        results = []
        for email_id, subject, body in cursor.fetchall():
            cache_path = Path(cache_dir) / f"{email_id}.ok"
            if not cache_path.exists():
                text = f"{subject or ''}\n\n{body or ''}".strip()
                results.append((email_id, text))
        
        return results

def fetch_emails_by_ids(self, ids: List[str]) -> List[Dict]:
    """Fetch email details by IDs in specified order."""
    if not ids:
        return []
    
    placeholders = ",".join("?" for _ in ids)
    with self.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT id, subject, from_addr, sent_at, body 
            FROM emails 
            WHERE id IN ({placeholders})
        """, ids)
        
        # Create lookup and maintain order
        email_map = {
            row[0]: {
                "id": row[0], "subject": row[1], "from": row[2], 
                "sent_at": row[3], "body": row[4]
            }
            for row in cursor.fetchall()
        }
        
        return [email_map[id_] for id_ in ids if id_ in email_map]
```

# ============ VECTOR INDEX ============

@dataclass
class VectorIndex:
“”“FAISS vector index with metadata.”””
index: faiss.IndexFlatIP
ids: List[str]
dimension: int

class EmbeddingManager:
“”“Manages embeddings and vector search index.”””

```
def __init__(self, config: Config, logger: logging.Logger):
    self.config = config
    self.logger = logger
    self._model = None

@property
def model(self) -> SentenceTransformer:
    """Lazy load the embedding model."""
    if self._model is None:
        self.logger.info(f"Loading embedding model: {self.config.embed_model}")
        self._model = SentenceTransformer(self.config.embed_model)
    return self._model

def load_or_create_index(self) -> VectorIndex:
    """Load existing index or create new one."""
    faiss_path = Path(self.config.faiss_path)
    map_path = Path(self.config.faiss_map_path)
    
    if faiss_path.exists() and map_path.exists():
        try:
            index = faiss.read_index(str(faiss_path))
            with open(map_path, 'r', encoding='utf-8') as f:
                ids = json.load(f)
            
            dimension = self.model.get_sentence_embedding_dimension()
            if index.d != dimension:
                self.logger.warning("Index dimension mismatch, creating new index")
                return self._create_new_index(dimension)
            
            return VectorIndex(index=index, ids=ids, dimension=dimension)
        except Exception as e:
            self.logger.warning(f"Failed to load index: {e}, creating new one")
    
    dimension = self.model.get_sentence_embedding_dimension()
    return self._create_new_index(dimension)

def _create_new_index(self, dimension: int) -> VectorIndex:
    """Create a new FAISS index."""
    return VectorIndex(
        index=faiss.IndexFlatIP(dimension),
        ids=[],
        dimension=dimension
    )

def save_index(self, vector_index: VectorIndex):
    """Save index to disk."""
    faiss.write_index(vector_index.index, self.config.faiss_path)
    with open(self.config.faiss_map_path, 'w', encoding='utf-8') as f:
        json.dump(vector_index.ids, f)

def add_embeddings(self, vector_index: VectorIndex, texts_and_ids: List[Tuple[str, str]]):
    """Add embeddings to index in batches."""
    if not texts_and_ids:
        return
    
    texts, ids = zip(*texts_and_ids)
    batch_size = self.config.batch_embed
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        end_idx = min(i + batch_size, len(texts))
        batch_texts = texts[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        # Generate embeddings
        embeddings = self.model.encode(
            batch_texts,
            show_progress_bar=False,
            normalize_embeddings=True
        ).astype('float32')
        
        # Add to index
        vector_index.index.add(embeddings)
        vector_index.ids.extend(batch_ids)
        
        # Mark as cached
        for email_id in batch_ids:
            cache_file = Path(self.config.cache_dir) / f"{email_id}.ok"
            cache_file.touch()

def search(self, vector_index: VectorIndex, query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Search for similar emails using vector similarity."""
    if len(vector_index.ids) == 0:
        return []
    
    # Create query embedding
    query_vector = self.model.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True
    ).astype('float32')
    
    # Search index
    search_k = min(k * 3, max(20, k))  # Get more results for filtering
    scores, indices = vector_index.index.search(query_vector, search_k)
    
    # Convert to results
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= idx < len(vector_index.ids):
            results.append((vector_index.ids[idx], float(score)))
    
    return results
```

# ============ OUTLOOK INTEGRATION ============

class OutlookCollector:
“”“Handles email collection from local Outlook.”””

```
MAIL_ITEM_CLASS = 43  # Outlook MailItem class constant

def __init__(self, config: Config, logger: logging.Logger):
    self.config = config
    self.logger = logger
    self.text_processor = TextProcessor()

def _get_outlook_namespace(self):
    """Get Outlook namespace with error handling."""
    try:
        import win32com.client as win32
        return win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
    except ImportError:
        raise RuntimeError("pywin32 package required for Outlook integration")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Outlook: {e}")

def list_folders(self):
    """List all available Outlook folders."""
    namespace = self._get_outlook_namespace()
    
    for i in range(1, namespace.Folders.Count + 1):
        store = namespace.Folders.Item(i)
        print(f"\n=== STORE: {store.Name} ===")
        self._print_folder_tree(store, "")

def _print_folder_tree(self, folder, prefix: str):
    """Recursively print folder structure."""
    try:
        print(f"{prefix}{folder.Name}")
        for i in range(1, folder.Folders.Count + 1):
            subfolder = folder.Folders.Item(i)
            self._print_folder_tree(subfolder, f"{prefix}{folder.Name}\\")
    except Exception:
        pass  # Some folders may not be accessible

def _resolve_folder(self) -> object:
    """Resolve the target Outlook folder."""
    namespace = self._get_outlook_namespace()
    
    # Find the right store
    target_store = None
    if self.config.outlook_store_hint:
        hint_lower = self.config.outlook_store_hint.lower()
        for i in range(1, namespace.Folders.Count + 1):
            store = namespace.Folders.Item(i)
            if hint_lower in str(store.Name).lower():
                target_store = store
                break
    
    if target_store is None:
        # Use default store
        target_store = namespace.GetDefaultFolder(6).Parent
    
    # Navigate to the specified folder path
    folder = target_store
    path_parts = [p for p in self.config.outlook_folder_path.split("\\") if p]
    
    # Handle special case for Inbox
    if path_parts and path_parts[0].lower() == "inbox":
        try:
            folder = target_store.Store.GetDefaultFolder(6)
            path_parts = path_parts[1:]
        except Exception:
            folder = target_store.Folders["Inbox"]
            path_parts = path_parts[1:]
    
    # Navigate remaining path
    for part in path_parts:
        folder = folder.Folders[part]
    
    return folder

def collect_emails(self) -> Generator[Dict, None, None]:
    """Collect emails from Outlook folder."""
    try:
        folder = self._resolve_folder()
    except Exception as e:
        error_msg = (
            f"Could not resolve folder '{self.config.outlook_folder_path}' "
            f"in store '{self.config.outlook_store_hint}'. "
            f"Run with 'list-folders' command to see available paths. "
            f"Error: {e}"
        )
        raise RuntimeError(error_msg)
    
    items = folder.Items
    items.Sort("[ReceivedTime]", True)  # Sort by received time, newest first
    
    count = 0
    for mail in items:
        if count >= self.config.max_to_ingest:
            break
        
        try:
            # Only process mail items
            if getattr(mail, "Class", None) != self.MAIL_ITEM_CLASS:
                continue
            
            email_data = self._extract_email_data(mail)
            if email_data:
                yield email_data
                count += 1
                
        except Exception as e:
            self.logger.warning(f"Failed to process email: {e}")
            continue

def _extract_email_data(self, mail) -> Optional[Dict]:
    """Extract data from Outlook mail item."""
    try:
        message_id = str(getattr(mail, "InternetMessageID", "") or "")
        subject = str(getattr(mail, "Subject", "") or "")
        sender = str(getattr(mail, "SenderEmailAddress", "") or "")
        to_addrs = str(getattr(mail, "To", "") or "")
        cc_addrs = str(getattr(mail, "CC", "") or "")
        
        # Handle received time
        received_time = getattr(mail, "ReceivedTime", None)
        sent_at = received_time.strftime("%Y-%m-%dT%H:%M:%SZ") if received_time else None
        
        # Extract body content
        html_body = str(getattr(mail, "HTMLBody", "") or "")
        text_body = str(getattr(mail, "Body", "") or "")
        
        body = self.text_processor.html_to_text(html_body) if html_body else text_body
        body = clean_text(body)
        
        # Create stable ID
        id_source = message_id.strip() or f"{subject}{sender}{sent_at or ''}".lower()
        stable_id = message_id.strip() or compute_hash(id_source)
        
        return {
            "id": stable_id,
            "message_id": message_id,
            "subject": subject,
            "from_addr": sender,
            "to_addrs": to_addrs,
            "cc_addrs": cc_addrs,
            "sent_at": sent_at,
            "folder": self.config.outlook_folder_path,
            "headers": json.dumps({"internet_message_id": message_id}),
            "body": body
        }
        
    except Exception as e:
        self.logger.warning(f"Failed to extract email data: {e}")
        return None
```

# ============ MAIN PIPELINE ============

class EmailArchiveSystem:
“”“Main system coordinator.”””

```
def __init__(self, config: Config, logger: logging.Logger):
    self.config = config
    self.logger = logger
    
    self.database = EmailDatabase(config.db_path, logger)
    self.embedding_manager = EmbeddingManager(config, logger)
    self.outlook_collector = OutlookCollector(config, logger)

def ingest(self):
    """Ingest emails from Outlook and update search index."""
    self.logger.info("Starting email ingestion...")
    
    # Collect emails from Outlook
    emails = list(self.outlook_collector.collect_emails())
    
    if not emails:
        self.logger.info("No new emails found")
        return
    
    # Store in database
    inserted_count = self.database.upsert_emails(emails)
    self.logger.info(f"Processed {len(emails)} emails, {inserted_count} new")
    
    # Update embeddings
    self._update_embeddings()

def _update_embeddings(self):
    """Update vector embeddings for new emails."""
    # Get emails needing embeddings
    pending = self.database.get_emails_needing_vectors(self.config.cache_dir)
    
    if not pending:
        self.logger.info("No new emails need embeddings")
        return
    
    self.logger.info(f"Creating embeddings for {len(pending)} emails...")
    
    # Load or create index
    vector_index = self.embedding_manager.load_or_create_index()
    
    # Add new embeddings
    self.embedding_manager.add_embeddings(vector_index, pending)
    
    # Save updated index
    self.embedding_manager.save_index(vector_index)
    
    self.logger.info(f"Index updated. Total vectors: {len(vector_index.ids)}")

def query(self, query_text: str, k: int = 10, since: Optional[str] = None, 
          before: Optional[str] = None):
    """Search emails using semantic similarity."""
    # Load index
    vector_index = self.embedding_manager.load_or_create_index()
    
    if len(vector_index.ids) == 0:
        print("No emails indexed. Run 'ingest' command first.")
        return
    
    # Search for similar emails
    search_results = self.embedding_manager.search(vector_index, query_text, k * 3)
    
    # Get email details
    email_ids = [result[0] for result in search_results]
    emails = self.database.fetch_emails_by_ids(email_ids)
    
    # Apply date filtering
    filtered_results = self._filter_by_date(search_results, emails, since, before)[:k]
    
    # Display results
    self._display_results(query_text, filtered_results, emails)

def _filter_by_date(self, search_results: List[Tuple[str, float]], 
                   emails: List[Dict], since: Optional[str], 
                   before: Optional[str]) -> List[Tuple[str, float]]:
    """Filter search results by date range."""
    if not since and not before:
        return search_results
    
    email_map = {email["id"]: email for email in emails}
    filtered = []
    
    for email_id, score in search_results:
        email = email_map.get(email_id)
        if not email or not email.get("sent_at"):
            continue
        
        try:
            sent_date = datetime.fromisoformat(
                email["sent_at"].replace("Z", "").replace("+00:00", "")
            )
            
            if since:
                since_date = datetime.fromisoformat(since)
                if sent_date < since_date:
                    continue
            
            if before:
                before_date = datetime.fromisoformat(before)
                if sent_date >= before_date:
                    continue
            
            filtered.append((email_id, score))
            
        except ValueError:
            continue
    
    return filtered

def _display_results(self, query: str, results: List[Tuple[str, float]], 
                    emails: List[Dict]):
    """Display search results in a formatted manner."""
    email_map = {email["id"]: email for email in emails}
    
    print(f"\nTop {len(results)} results for: {query}\n")
    
    for rank, (email_id, score) in enumerate(results, 1):
        email = email_map.get(email_id)
        if not email:
            continue
        
        sent_at = email.get("sent_at", "unknown")
        subject = email.get("subject") or "(no subject)"
        from_addr = email.get("from", "unknown")
        body_preview = preview_text(email.get("body", ""), self.config.preview_limit)
        
        print(f"{rank:>2}. score={score:.3f} | {sent_at}")
        print(f"    Subject: {subject}")
        print(f"    From:    {from_addr}")
        print(f"    Preview: {body_preview}\n")
```

# ============ CLI INTERFACE ============

def create_parser() -> argparse.ArgumentParser:
“”“Create command line argument parser.”””
parser = argparse.ArgumentParser(
description=“Local Email Archive System with Semantic Search”,
formatter_class=argparse.RawDescriptionHelpFormatter
)

```
parser.add_argument(
    "--config", 
    type=str, 
    help="Path to configuration file (JSON)"
)
parser.add_argument(
    "--log-level", 
    choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
    default="INFO",
    help="Logging level"
)

subparsers = parser.add_subparsers(dest="command", required=True)

# Ingest command
subparsers.add_parser(
    "ingest", 
    help="Collect emails from Outlook and update search index"
)

# List folders command
subparsers.add_parser(
    "list-folders", 
    help="List available Outlook stores and folders"
)

# Query command
query_parser = subparsers.add_parser(
    "query", 
    help="Search emails using semantic similarity"
)
query_parser.add_argument("text", help="Search query text")
query_parser.add_argument("-k", type=int, default=10, help="Number of results")
query_parser.add_argument("--since", help="Start date (YYYY-MM-DD)")
query_parser.add_argument("--before", help="End date (YYYY-MM-DD)")

return parser
```

def load_config(config_path: Optional[str] = None) -> Config:
“”“Load configuration from file or use defaults.”””
config_data = {}

```
if config_path and Path(config_path).exists():
    with open(config_path, 'r') as f:
        config_data = json.load(f)

return Config(**config_data)
```

def main():
“”“Main application entry point.”””
parser = create_parser()
args = parser.parse_args()

```
# Setup logging
logger = setup_logging(args.log_level)

try:
    # Load configuration
    config = load_config(args.config)
    
    # Create system
    system = EmailArchiveSystem(config, logger)
    
    # Execute command
    if args.command == "ingest":
        system.ingest()
    elif args.command == "list-folders":
        system.outlook_collector.list_folders()
    elif args.command == "query":
        system.query(args.text, args.k, args.since, args.before)
    
except KeyboardInterrupt:
    logger.info("Operation cancelled by user")
    sys.exit(1)
except Exception as e:
    logger.error(f"Application error: {e}")
    sys.exit(1)
```

if **name** == “**main**”:
main()
