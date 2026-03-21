#!/usr/bin/env python3
"""
Photo Move Executor - Processes CSV from photo_dedup_scanner.py to move unique photos.

This tool reads the CSV output from photo_dedup_scanner.py and performs the actual
file operations (copy/move) to organize photos into YYYY/MM/DD directory structure.

Features:
- Dry run mode to preview operations without making changes
- Summary statistics of all operations
- Skip existing files (default) - only adds new images not already in archive
- Force overwrite option to replace existing files
- Rename collision option to auto-rename if file exists
- Verification of copied files using SHA256 hash
- Detailed logging of all operations
- Does NOT delete source files (safe copy operation)

Usage:
    python photo_move_executor.py --input photo_plan.csv --archive /path/to/archive
    python photo_move_executor.py --input photo_plan.csv --archive /path/to/archive --dry-run
    python photo_move_executor.py --input photo_plan.csv --archive /path/to/archive --overwrite
    python photo_move_executor.py --input photo_plan.csv --archive /path/to/archive --rename-collision
"""

import argparse
import csv
import os
import sys
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class PhotoMoveExecutor:
    """Executes photo moves based on CSV plan from photo_dedup_scanner.py."""
    
    def __init__(self, archive_path: str, dry_run: bool = False, 
                 overwrite: bool = False, rename_collision: bool = False,
                 verbose: bool = False):
        self.archive_path = Path(archive_path).expanduser().resolve()
        self.dry_run = dry_run
        self.overwrite = overwrite
        self.rename_collision = rename_collision
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            'total_rows': 0,
            'skipped_duplicates': 0,
            'skipped_existing': 0,
            'copied': 0,
            'overwritten': 0,
            'renamed': 0,
            'errors': 0,
            'verification_failures': 0,
            'total_bytes': 0,
        }
        
        # Operation log for detailed reporting
        self.operations: List[Dict] = []
    
    def calculate_sha256(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def get_unique_filename(self, target_dir: Path, base_filename: str) -> str:
        """Generate a unique filename if collision exists."""
        base_name = Path(base_filename).stem
        extension = Path(base_filename).suffix
        
        counter = 1
        new_filename = base_filename
        
        while (target_dir / new_filename).exists():
            new_filename = f"{base_name}_{counter:03d}{extension}"
            counter += 1
        
        return new_filename
    
    def verify_copy(self, source: Path, dest: Path) -> bool:
        """Verify that copied file matches source using SHA256."""
        try:
            source_hash = self.calculate_sha256(source)
            dest_hash = self.calculate_sha256(dest)
            return source_hash == dest_hash
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Verification error: {e}")
            return False
    
    def process_row(self, row: Dict) -> Optional[Dict]:
        """Process a single row from the CSV."""
        filepath = Path(row['filepath']).expanduser().resolve()
        target_dir = self.archive_path / row['target_directory']
        new_filename = row['new_filename']
        is_duplicate = row.get('is_duplicate', 'False').lower() == 'true'
        expected_sha256 = row.get('sha256', '')
        
        # Skip duplicates
        if is_duplicate:
            self.stats['skipped_duplicates'] += 1
            return {
                'action': 'skipped_duplicate',
                'source': str(filepath),
                'target': str(target_dir / new_filename),
                'reason': 'Exact duplicate (same SHA256)'
            }
        
        # Check if source file exists
        if not filepath.exists():
            self.stats['errors'] += 1
            return {
                'action': 'error',
                'source': str(filepath),
                'target': str(target_dir / new_filename),
                'reason': 'Source file does not exist'
            }
        
        # Create target directory if needed
        if not self.dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = target_dir / new_filename
        
        # Handle existing files in target
        if target_path.exists():
            if self.overwrite:
                action = 'overwritten'
                self.stats['overwritten'] += 1
            elif self.rename_collision:
                new_filename = self.get_unique_filename(target_dir, new_filename)
                target_path = target_dir / new_filename
                action = 'renamed'
                self.stats['renamed'] += 1
            else:
                # Default: skip existing
                self.stats['skipped_existing'] += 1
                return {
                    'action': 'skipped_existing',
                    'source': str(filepath),
                    'target': str(target_path),
                    'reason': 'File already exists (use --overwrite or --rename-collision)'
                }
        else:
            action = 'copied'
            self.stats['copied'] += 1
        
        # Perform the copy
        if not self.dry_run:
            try:
                shutil.copy2(filepath, target_path)  # copy2 preserves metadata
                
                # Verify the copy
                if not self.verify_copy(filepath, target_path):
                    self.stats['verification_failures'] += 1
                    # Try to clean up failed copy
                    if target_path.exists():
                        target_path.unlink()
                    return {
                        'action': 'verification_failed',
                        'source': str(filepath),
                        'target': str(target_path),
                        'reason': 'SHA256 verification failed after copy'
                    }
                
                file_size = filepath.stat().st_size
                self.stats['total_bytes'] += file_size
                
            except Exception as e:
                self.stats['errors'] += 1
                return {
                    'action': 'error',
                    'source': str(filepath),
                    'target': str(target_path),
                    'reason': str(e)
                }
        
        return {
            'action': action,
            'source': str(filepath),
            'target': str(target_path),
            'size': filepath.stat().st_size if not self.dry_run else None
        }
    
    def process_csv(self, csv_path: str) -> None:
        """Process the entire CSV file."""
        csv_path = Path(csv_path).expanduser().resolve()
        
        if not csv_path.exists():
            print(f"❌ Error: CSV file not found: {csv_path}")
            sys.exit(1)
        
        print(f"📋 Processing CSV: {csv_path}")
        print(f"📁 Archive directory: {self.archive_path}")
        print(f"🔍 Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"📝 Options: overwrite={self.overwrite}, rename_collision={self.rename_collision}")
        print("-" * 80)
        
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                self.stats['total_rows'] += 1
                
                if self.verbose:
                    print(f"\n[{row_num}] Processing: {row['filename']}")
                
                result = self.process_row(row)
                
                if result:
                    self.operations.append(result)
                    
                    if self.verbose:
                        action = result['action'].replace('_', ' ').title()
                        print(f"  → {action}: {result['target']}")
                        if 'reason' in result:
                            print(f"     Reason: {result['reason']}")
                
                # Progress indicator for non-verbose mode
                if not self.verbose and row_num % 100 == 0:
                    print(f"  Processed {row_num} rows...")
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("📊 EXECUTION SUMMARY")
        print("=" * 80)
        
        print(f"\n📋 Total rows processed:     {self.stats['total_rows']}")
        print(f"\n✅ Successful Operations:")
        print(f"   • Files copied:           {self.stats['copied']}")
        print(f"   • Files overwritten:      {self.stats['overwritten']}")
        print(f"   • Files renamed:          {self.stats['renamed']}")
        
        print(f"\n⏭️  Skipped:")
        print(f"   • Duplicates skipped:     {self.stats['skipped_duplicates']}")
        print(f"   • Existing files skipped: {self.stats['skipped_existing']}")
        
        if self.stats['errors'] > 0 or self.stats['verification_failures'] > 0:
            print(f"\n⚠️  Issues:")
            print(f"   • Errors:                 {self.stats['errors']}")
            print(f"   • Verification failures:  {self.stats['verification_failures']}")
        
        if self.stats['total_bytes'] > 0:
            size_mb = self.stats['total_bytes'] / (1024 * 1024)
            size_gb = self.stats['total_bytes'] / (1024 * 1024 * 1024)
            print(f"\n💾 Total data transferred:  {size_mb:.2f} MB ({size_gb:.3f} GB)")
        
        if self.dry_run:
            print("\n🔍 This was a DRY RUN. No files were actually copied.")
            print("   Run without --dry-run to perform the actual operations.")
        else:
            print("\n✅ Operations completed successfully!")
        
        # Save detailed log if requested
        if self.verbose and self.operations:
            log_file = Path('move_operations_log.txt')
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("Photo Move Operations Log\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Archive: {self.archive_path}\n")
                f.write(f"Dry Run: {self.dry_run}\n")
                f.write("=" * 80 + "\n\n")
                
                for op in self.operations:
                    f.write(f"[{op['action'].upper()}]\n")
                    f.write(f"  Source: {op['source']}\n")
                    f.write(f"  Target: {op['target']}\n")
                    if 'reason' in op:
                        f.write(f"  Reason: {op['reason']}\n")
                    if 'size' in op and op['size']:
                        f.write(f"  Size: {op['size']} bytes\n")
                    f.write("\n")
            
            print(f"\n📝 Detailed log saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Execute photo moves based on CSV plan from photo_dedup_scanner.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - preview operations without making changes
  python photo_move_executor.py --input photo_plan.csv --archive ~/photo_archive --dry-run
  
  # Live run - only add new files (skip existing)
  python photo_move_executor.py --input photo_plan.csv --archive ~/photo_archive
  
  # Overwrite existing files in archive
  python photo_move_executor.py --input photo_plan.csv --archive ~/photo_archive --overwrite
  
  # Auto-rename files that would collide
  python photo_move_executor.py --input photo_plan.csv --archive ~/photo_archive --rename-collision
  
  # Verbose output with detailed logging
  python photo_move_executor.py --input photo_plan.csv --archive ~/photo_archive --verbose
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file from photo_dedup_scanner.py')
    parser.add_argument('--archive', '-a', required=True,
                       help='Target archive directory (YYYY/MM/DD structure will be created)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Preview operations without making changes')
    parser.add_argument('--overwrite', '-o', action='store_true',
                       help='Overwrite existing files in archive (default: skip)')
    parser.add_argument('--rename-collision', '-r', action='store_true',
                       help='Auto-rename files that would collide (adds _001, _002, etc.)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with per-file details')
    
    args = parser.parse_args()
    
    # Validate options
    if args.overwrite and args.rename_collision:
        print("⚠️  Warning: Both --overwrite and --rename-collision specified.")
        print("   --overwrite will take precedence.")
    
    executor = PhotoMoveExecutor(
        archive_path=args.archive,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        rename_collision=args.rename_collision,
        verbose=args.verbose
    )
    
    executor.process_csv(args.input)


if __name__ == '__main__':
    main()
