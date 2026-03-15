#!/usr/bin/env python3
"""Image Organization Utility - Organizes images by date and provides search."""
import os, sys, json, shutil, sqlite3, hashlib, re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_IMAGE_SUPPORT = True
except ImportError as e:
    print(f"Warning: Image libraries not installed: {e}")
    HAS_IMAGE_SUPPORT = False

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}

@dataclass
class OrganizedImageInfo:
    file_path: str; original_path: str; organized_path: str; file_size: int
    checksum_sha256: str; dimensions: Tuple[int, int]; format: str
    exif_date: Optional[str]; parsed_date: Optional[str]
    year: Optional[int]; month: Optional[int]; day: Optional[int]
    gps_latitude: Optional[float]; gps_longitude: Optional[float]
    camera_make: Optional[str]; camera_model: Optional[str]
    caption: Optional[str]; tags: List[str]
    def to_dict(self) -> dict:
        d = asdict(self); d['dimensions'] = list(d['dimensions']); return d
    @classmethod
    def from_dict(cls, data: dict):
        if 'dimensions' in data and isinstance(data['dimensions'], list):
            data['dimensions'] = tuple(data['dimensions'])
        return cls(**data)

def parse_exif_date(date_str: str) -> Optional[datetime]:
    if not date_str: return None
    for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y:%m:%d', '%Y-%m-%d']:
        try: return datetime.strptime(date_str, fmt)
        except ValueError: continue
    return None

def convert_gps_coordinate(value, ref):
    """Convert GPS EXIF data to decimal degrees"""
    if not value or len(value) < 3:
        return None
    try:
        # GPS coordinates are stored as degrees, minutes, seconds
        degrees = float(value[0]) if isinstance(value[0], (int, float)) else float(value[0].num) / float(value[0].den)
        minutes = float(value[1]) if isinstance(value[1], (int, float)) else float(value[1].num) / float(value[1].den)
        seconds = float(value[2]) if isinstance(value[2], (int, float)) else float(value[2].num) / float(value[2].den)
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal
    except (Exception, AttributeError):
        return None

def extract_metadata(file_path: Path) -> dict:
    metadata = {'exif_date': None, 'parsed_date': None, 'year': None, 'month': None, 'day': None,
                'gps_latitude': None, 'gps_longitude': None, 'camera_make': None, 'camera_model': None}
    if not HAS_IMAGE_SUPPORT:
        dt = datetime.fromtimestamp(os.path.getmtime(file_path))
        metadata['parsed_date'], metadata['year'], metadata['month'], metadata['day'] = dt.isoformat(), dt.year, dt.month, dt.day
        return metadata
    try:
        with Image.open(file_path) as img:
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                TAGS = {271: 'make', 272: 'model', 306: 'datetime', 36867: 'datetime_original',
                        34853: 'gps_info'}
                exif_data = {TAGS.get(tid, tid): v for tid, v in exif.items()}
                
                # Extract date information
                date_str = exif_data.get('datetime_original') or exif_data.get('datetime')
                if date_str:
                    metadata['exif_date'] = date_str
                    parsed = parse_exif_date(date_str)
                    if parsed:
                        metadata['parsed_date'] = parsed.isoformat()
                        metadata['year'], metadata['month'], metadata['day'] = parsed.year, parsed.month, parsed.day
                metadata['camera_make'], metadata['camera_model'] = exif_data.get('make'), exif_data.get('model')
                
                # Extract GPS information
                gps_info = exif_data.get('gps_info') or exif.get(34853)
                if gps_info:
                    # GPS tags: 1=lat_ref, 2=latitude, 3=lon_ref, 4=longitude
                    lat_ref = gps_info.get(1)  # 'N' or 'S'
                    lat = gps_info.get(2)      # tuple of (deg, min, sec)
                    lon_ref = gps_info.get(3)  # 'E' or 'W'
                    lon = gps_info.get(4)      # tuple of (deg, min, sec)
                    
                    if lat and lat_ref:
                        metadata['gps_latitude'] = convert_gps_coordinate(lat, lat_ref)
                    if lon and lon_ref:
                        metadata['gps_longitude'] = convert_gps_coordinate(lon, lon_ref)
                        
            if not metadata['parsed_date']:
                dt = datetime.fromtimestamp(os.path.getmtime(file_path))
                metadata['parsed_date'], metadata['year'], metadata['month'], metadata['day'] = dt.isoformat(), dt.year, dt.month, dt.day
    except Exception as e:
        print(f"Warning: Could not extract metadata from {file_path}: {e}")
    return metadata

def calculate_checksum(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
    return h.hexdigest()

def organize_by_date(source_dir: Path, dest_dir: Path, move: bool = False, dry_run: bool = False) -> List[OrganizedImageInfo]:
    organized_images = []
    if not source_dir.exists(): print(f"Error: Source directory does not exist: {source_dir}"); return organized_images
    files = list(source_dir.glob('**/*'))
    image_files = [f for f in files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    print(f"Found {len(image_files)} images to organize")
    for i, file_path in enumerate(image_files, 1):
        if i % 10 == 0: print(f"  Processing {i}/{len(image_files)}...")
        try:
            meta = extract_metadata(file_path)
            if meta['year'] and meta['month'] and meta['day']: rel_path = f"{meta['year']}/{meta['month']:02d}/{meta['day']:02d}"
            elif meta['year'] and meta['month']: rel_path = f"{meta['year']}/{meta['month']:02d}/unknown_day"
            elif meta['year']: rel_path = f"{meta['year']}/unknown_month"
            else: rel_path = "unknown_date"
            dest_subdir, dest_file = dest_dir / rel_path, dest_dir / rel_path / file_path.name
            if dest_file.exists():
                cs = calculate_checksum(file_path)[:8]
                dest_file = dest_subdir / f"{file_path.stem}_{cs}{file_path.suffix}"
            if not dry_run:
                dest_subdir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(dest_file)) if move else shutil.copy2(str(file_path), str(dest_file))
            dims, fmt = (0, 0), 'UNKNOWN'
            if HAS_IMAGE_SUPPORT:
                try:
                    with Image.open(file_path) as img: dims, fmt = img.size, img.format or 'UNKNOWN'
                except: pass
            organized_images.append(OrganizedImageInfo(
                file_path=str(dest_file.absolute()) if not dry_run else str(dest_file),
                original_path=str(file_path.absolute()), organized_path=str(dest_file),
                file_size=file_path.stat().st_size, checksum_sha256=calculate_checksum(file_path),
                dimensions=dims, format=fmt, exif_date=meta['exif_date'], parsed_date=meta['parsed_date'],
                year=meta['year'], month=meta['month'], day=meta['day'],
                gps_latitude=meta['gps_latitude'], gps_longitude=meta['gps_longitude'],
                camera_make=meta['camera_make'], camera_model=meta['camera_model'], caption=None, tags=[]
            ))
            if dry_run: print(f"  {'Would move' if move else 'Would copy'}: {file_path} -> {dest_file}")
        except Exception as e: print(f"Error processing {file_path}: {e}")
    print(f"\n{'Dry run' if dry_run else 'Organization'} complete. {len(organized_images)} images.")
    return organized_images

class ImageDatabase:
    def __init__(self, db_path: Path):
        self.db_path, self.conn = db_path, sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row; self._create_tables()
    def _create_tables(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, file_path TEXT UNIQUE NOT NULL,
            original_path TEXT, checksum_sha256 TEXT NOT NULL, file_size INTEGER, width INTEGER, height INTEGER, format TEXT,
            exif_date TEXT, parsed_date TEXT, year INTEGER, month INTEGER, day INTEGER, gps_latitude REAL, gps_longitude REAL,
            camera_make TEXT, camera_model TEXT, caption TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS image_tags (image_id INTEGER, tag_id INTEGER, PRIMARY KEY (image_id, tag_id),
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE, FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_images_date ON images(year, month, day)'); self.conn.commit()
    def add_image(self, img: OrganizedImageInfo) -> int:
        c = self.conn.cursor()
        c.execute('''INSERT OR REPLACE INTO images (file_path, original_path, checksum_sha256, file_size, width, height, format,
            exif_date, parsed_date, year, month, day, gps_latitude, gps_longitude, camera_make, camera_model, caption)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (img.file_path, img.original_path, img.checksum_sha256, img.file_size, img.dimensions[0], img.dimensions[1],
             img.format, img.exif_date, img.parsed_date, img.year, img.month, img.day, img.gps_latitude, img.gps_longitude,
             img.camera_make, img.camera_model, img.caption)); self.conn.commit(); return c.lastrowid
    def add_images(self, images: List[OrganizedImageInfo]) -> int:
        count = 0
        for img in images:
            try: self.add_image(img); count += 1
            except Exception as e: print(f"Error adding {img.file_path}: {e}")
        return count
    def update_caption(self, image_id: int, caption: str):
        c = self.conn.cursor(); c.execute('UPDATE images SET caption = ? WHERE id = ?', (caption, image_id)); self.conn.commit()
    def add_tag_to_image(self, image_id: int, tag_name: str):
        c = self.conn.cursor()
        c.execute('INSERT OR IGNORE INTO tags (name) VALUES (?)', (tag_name,))
        c.execute('SELECT id FROM tags WHERE name = ?', (tag_name,)); tag_id = c.fetchone()[0]
        c.execute('INSERT OR IGNORE INTO image_tags (image_id, tag_id) VALUES (?, ?)', (image_id, tag_id)); self.conn.commit()
    def search(self, query: str) -> List[dict]:
        c, year_matches = self.conn.cursor(), re.findall(r'\b(20\d{2})\b', query)
        month_patterns = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
                          'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        month_match = next((m for n,m in month_patterns.items() if n in query.lower()), None)
        text_query = query
        for y in year_matches: text_query = text_query.replace(y, '')
        if month_match:
            for n in month_patterns.keys(): text_query = re.sub(r'\b'+n+r'\b','',text_query,flags=re.IGNORECASE)
        text_query = text_query.strip()
        conditions, params = [], []
        if text_query:
            for word in text_query.split():
                if len(word) > 2: conditions.append('(caption LIKE ? OR camera_make LIKE ? OR camera_model LIKE ?)'); params.extend([f'%{word}%']*3)
        for y in year_matches: conditions.append('year = ?'); params.append(int(y))
        if month_match: conditions.append('month = ?'); params.append(month_match)
        sql = 'SELECT * FROM images WHERE '+' AND '.join(conditions) if conditions else 'SELECT * FROM images LIMIT 100'
        c.execute(sql, params); return [dict(row) for row in c.fetchall()]
    def get_stats(self) -> dict:
        c = self.conn.cursor()
        c.execute('SELECT COUNT(*) FROM images'); stats = {'total_images': c.fetchone()[0]}
        c.execute('SELECT COUNT(DISTINCT checksum_sha256) FROM images'); stats['unique_images'] = c.fetchone()[0]
        c.execute('SELECT MIN(year), MAX(year) FROM images WHERE year IS NOT NULL'); r=c.fetchone(); stats['year_range']=(r[0],r[1]) if r[0] else None
        return stats
    def close(self): self.conn.close()

def main():
    import argparse
    p = argparse.ArgumentParser(description='Image Organization Utility')
    sp = p.add_subparsers(dest='command')
    o = sp.add_parser('organize'); o.add_argument('--source',required=True); o.add_argument('--dest',required=True)
    o.add_argument('--move',action='store_true'); o.add_argument('--dry-run',action='store_true'); o.add_argument('--save-db')
    o.add_argument('--postgres-db', help='PostgreSQL connection string for storing metadata')
    i = sp.add_parser('index'); i.add_argument('--dir',required=True); i.add_argument('--output',required=True)
    s = sp.add_parser('search'); s.add_argument('--db',required=True); s.add_argument('--query',required=True); s.add_argument('--limit',type=int,default=20)
    s.add_argument('--postgres-db', help='PostgreSQL connection string')
    st = sp.add_parser('stats'); st.add_argument('--db',required=True); st.add_argument('--postgres-db')
    args = p.parse_args()
    if args.command == 'organize':
        org = organize_by_date(Path(args.source), Path(args.dest), args.move, args.dry_run)
        if args.save_db and not args.dry_run:
            db = ImageDatabase(Path(args.save_db)); cnt = db.add_images(org); print(f"Added {cnt} images"); db.close()
        # Also save to PostgreSQL if provided
        if args.postgres_db and not args.dry_run:
            try:
                from image_database import ImageDatabase as PostgresDB, ImageMetadata
                pgdb = PostgresDB(args.postgres_db)
                metadata_list = []
                for img in org:
                    # Convert OrganizedImageInfo to ImageMetadata
                    date_created = datetime.fromisoformat(img.parsed_date) if img.parsed_date else None
                    date_modified = datetime.fromtimestamp(os.path.getmtime(img.file_path))
                    # Calculate perceptual hash
                    phash = ""
                    if HAS_IMAGEHASH:
                        try:
                            with Image.open(img.file_path) as im:
                                phash = str(imagehash.phash(im))
                        except: pass
                    meta = ImageMetadata(
                        file_path=img.file_path,
                        file_name=os.path.basename(img.file_path),
                        file_size=img.file_size,
                        sha256=img.checksum_sha256,
                        perceptual_hash=phash,
                        width=img.dimensions[0] if img.dimensions else 0,
                        height=img.dimensions[1] if img.dimensions else 0,
                        format=img.format or 'UNKNOWN',
                        date_created=date_created,
                        date_modified=date_modified,
                        gps_latitude=img.gps_latitude,
                        gps_longitude=img.gps_longitude,
                        caption=None,
                        tags=[]
                    )
                    metadata_list.append(meta)
                count = pgdb.batch_insert_images(metadata_list)
                print(f"Added {count} images to PostgreSQL database")
                pgdb.close()
            except Exception as e:
                print(f"Warning: Could not save to PostgreSQL: {e}")
    elif args.command == 'index':
        dp, dbp = Path(args.dir), Path(args.output)
        if not dp.exists(): print(f"Error: {dp} does not exist"); sys.exit(1)
        imgs = [f for f in dp.glob('**/*') if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
        print(f"Found {len(imgs)} images to index"); organized = []
        for i, fp in enumerate(imgs, 1):
            if i % 10 == 0: print(f"  Indexing {i}/{len(imgs)}...")
            try:
                m = extract_metadata(fp); dims, fmt = (0,0), 'UNKNOWN'
                if HAS_IMAGE_SUPPORT:
                    try:
                        with Image.open(fp) as im: dims, fmt = im.size, im.format or 'UNKNOWN'
                    except: pass
                organized.append(OrganizedImageInfo(file_path=str(fp.absolute()),original_path=str(fp.absolute()),organized_path=str(fp.absolute()),
                    file_size=fp.stat().st_size,checksum_sha256=calculate_checksum(fp),dimensions=dims,format=fmt,exif_date=m['exif_date'],
                    parsed_date=m['parsed_date'],year=m['year'],month=m['month'],day=m['day'],gps_latitude=m['gps_latitude'],
                    gps_longitude=m['gps_longitude'],camera_make=m['camera_make'],camera_model=m['camera_model'],caption=None,tags=[]))
            except Exception as e: print(f"Error indexing {fp}: {e}")
        db = ImageDatabase(dbp); cnt = db.add_images(organized); st = db.get_stats()
        print(f"\nIndexing complete! Added {cnt} images, Year range: {st['year_range']}"); db.close()
    elif args.command == 'search':
        dbp = Path(args.db)
        if not dbp.exists(): print(f"Error: {dbp} does not exist"); sys.exit(1)
        db = ImageDatabase(dbp); results = db.search(args.query)
        print(f"\nSearch: '{args.query}' ({len(results)} found)\n")
        for i, img in enumerate(results[:args.limit], 1):
            print(f"{i}. {img['file_path']}")
            if img.get('caption'): print(f"   Caption: {img['caption']}")
            if img.get('parsed_date'): print(f"   Date: {img['parsed_date']}")
            if img.get('camera_model'): print(f"   Camera: {img['camera_model']}")
            print()
        if len(results) > args.limit: print(f"... and {len(results)-args.limit} more")
        db.close()
    elif args.command == 'stats':
        dbp = Path(args.db)
        if not dbp.exists(): print(f"Error: {dbp} does not exist"); sys.exit(1)
        db = ImageDatabase(dbp); st = db.get_stats()
        print("\n=== Database Statistics ==="); print(f"Total: {st['total_images']}, Unique: {st['unique_images']}")
        if st['year_range']: print(f"Year range: {st['year_range'][0]}-{st['year_range'][1]}"); db.close()
    else: p.print_help()

if __name__ == '__main__': main()
