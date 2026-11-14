import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from app.ingestion.indexer import ClinicalIndexer
from app.ingestion.config import MAX_WORKERS
from app.ingestion.pdf_worker import process_pdf

class IngestionManager:
    """Process PDFs concurrently and upload through one parent process."""
    
    def index_files(self,pdf_paths:Iterable[str | Path])->dict:
        paths=[Path(path).resolve() for path in pdf_paths]
        if not paths:
            raise ValueError("No pdf files were provided.")
        
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"PDF not found: {path}")
            if path.suffix.lower()!=".pdf":
                raise ValueError(f"Not a PDF file: {path}")
            
        results={}
        
        indexer=ClinicalIndexer()
        
        try:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures={
                    executor.submit(process_pdf,str(path)):path
                    for path in paths
                }
                for future in as_completed(futures):
                    path=futures[future]
                    key=str(path)
                    
                    try:
                        prepared=future.result()
                        if prepared["status"]!="success":
                            results[key]={
                                "status":"failed",
                                "error":prepared.get(
                                    "error","PDF processing failed."
                                )
                            }
                            continue
                        
                        chunk_count=indexer.index_chunks(prepared["chunks"])
                        results[key]={
                            "status":"success",
                            "chunks":chunk_count
                        }
                    except Exception as exc:
                        results[key]={
                            "status":"failed",
                            "error":str(exc)
                        }
        finally:
            indexer.close()
        
        return {
            "total_files":len(paths),
            "successful":sum(
                result["status"]=="success"
                for result in results.values()
            ),
            "failed":sum(
                result["status"]=="failed"
                for result in results.values()
            ),
            "files":results
        }
        
if __name__ == "__main__":
    parser=argparse.ArgumentParser(
        description="Multiprocess clinical PDF ingestion."
    )
    parser.add_argument(
        "pdf_paths",
        nargs="+",
        help="One or more PDF paths."
    )
    args=parser.parse_args()
    
    manager=IngestionManager()
    report=manager.index_files(args.pdf_paths)
    print(json.dumps(report,indent=2))