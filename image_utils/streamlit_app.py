import os
import io
import json
import psycopg2
import psycopg2.extras
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Photo Archive Explorer", layout="wide")


def connect_db(conn_str: str):
    try:
        # Close any previously-stored connection to avoid reusing an aborted transaction
        prev = st.session_state.get('conn')
        try:
            if prev:
                try:
                    prev.close()
                except Exception:
                    pass
        except Exception:
            pass

        conn = psycopg2.connect(conn_str)
        conn.autocommit = True
        return conn
    except Exception as e:
        st.error(f"Could not connect: {e}")
        return None


def get_stats(conn):
    q = {
        'total': "SELECT COUNT(*) FROM images",
        'with_captions': "SELECT COUNT(*) FROM images WHERE caption IS NOT NULL",
        'with_gps': "SELECT COUNT(*) FROM images WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL"
    }
    out = {}
    # Open a fresh connection for stats to avoid any aborted transaction state
    conn_str = st.session_state.get('conn_str')
    if not conn_str:
        st.warning('No connection string stored; connect first')
        return out
    try:
        with psycopg2.connect(conn_str) as local_conn:
            local_conn.autocommit = True
            with local_conn.cursor() as cur:
                for k, sql in q.items():
                    try:
                        cur.execute(sql)
                        out[k] = cur.fetchone()[0]
                    except Exception as e:
                        out[k] = None
                        msg = str(e)
                        # detect missing table
                        if 'relation "images" does not exist' in msg:
                            st.error('Database schema not initialized (images table missing).')
                            st.info('You can create the schema by running:')
                            st.code(f"python image_database.py init --db \"{conn_str}\"")
                        elif 'current transaction is aborted' in msg:
                            # clear any stored connection to avoid reuse
                            try:
                                prev = st.session_state.pop('conn', None)
                                if prev:
                                    try:
                                        prev.close()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            st.error('A previous DB error left a transaction aborted; connection reset. Reconnect and try again.')
                        else:
                            st.warning(f"Stats query failed ({k}): {e}")
    except Exception as e:
        st.error(f"Could not run stats queries: {e}")
    return out


def init_schema(conn_str: str):
    try:
        # Use existing ImageDatabase initializer to create schema
        from image_database import ImageDatabase
        db = ImageDatabase(conn_str)
        db.close()
        st.success('Database schema initialized (images table created).')
    except Exception as e:
        st.error(f'Failed to initialize schema: {e}')


def embed_query_local(model_name: str, text: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        st.warning("Local embedding unavailable. Install sentence-transformers to enable pgvector search.")
        return None
    model = SentenceTransformer(model_name)
    vec = model.encode([text])[0].tolist()
    return vec


def query_caption_pgvector(query_vec, limit=20):
    conn_str = st.session_state.get('conn_str')
    if not conn_str:
        st.error('No connection string stored; connect first')
        return []
    emb_str = '[' + ','.join(map(str, query_vec)) + ']'
    sql = """
    SELECT id, file_path, file_name, caption, gps_latitude, gps_longitude, width, height, format,
           1 - (caption_embedding <=> %s::vector) as similarity_score
    FROM images
    WHERE caption_embedding IS NOT NULL
    ORDER BY caption_embedding <=> %s::vector
    LIMIT %s
    """
    try:
        with psycopg2.connect(conn_str) as local_conn:
            local_conn.autocommit = True
            with local_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (emb_str, emb_str, limit))
                return cur.fetchall()
    except Exception as e:
        msg = str(e)
        if 'relation "images" does not exist' in msg:
            st.error('Database schema not initialized (images table missing). Click "Initialize schema" in the sidebar to create it.')
            st.info('Or run:')
            st.code(f"python image_database.py init --db \"{conn_str}\"")
            if st.sidebar.button('Initialize schema'):
                init_schema(conn_str)
        elif 'current transaction is aborted' in msg:
            try:
                prev = st.session_state.pop('conn', None)
                if prev:
                    try:
                        prev.close()
                    except Exception:
                        pass
            except Exception:
                pass
            st.error('A previous DB error left a transaction aborted; connection reset. Reconnect and try again.')
        else:
            st.error(f"pgvector query failed: {e}")
        return []


def query_caption_text(text_q, limit=50):
    conn_str = st.session_state.get('conn_str')
    if not conn_str:
        st.error('No connection string stored; connect first')
        return []
    pattern = f"%{text_q}%"
    sql = "SELECT id, file_path, file_name, caption, gps_latitude, gps_longitude, width, height, format FROM images WHERE caption ILIKE %s ORDER BY date_created DESC LIMIT %s"
    try:
        with psycopg2.connect(conn_str) as local_conn:
            local_conn.autocommit = True
            with local_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (pattern, limit))
                return cur.fetchall()
    except Exception as e:
        msg = str(e)
        if 'relation "images" does not exist' in msg:
            st.error('Database schema not initialized (images table missing). Click "Initialize schema" in the sidebar to create it.')
            st.info('Or run:')
            st.code(f"python image_database.py init --db \"{conn_str}\"")
            if st.sidebar.button('Initialize schema'):
                init_schema(conn_str)
        elif 'current transaction is aborted' in msg:
            try:
                prev = st.session_state.pop('conn', None)
                if prev:
                    try:
                        prev.close()
                    except Exception:
                        pass
            except Exception:
                pass
            st.error('A previous DB error left a transaction aborted; connection reset. Reconnect and try again.')
        else:
            st.error(f"Text query failed: {e}")
        return []


def query_metadata_location(lat, lon, radius_km=5.0, limit=100):
    conn_str = st.session_state.get('conn_str')
    if not conn_str:
        st.error('No connection string stored; connect first')
        return []
    sql = f"""
    SELECT id, file_path, file_name, caption, gps_latitude, gps_longitude, width, height, format,
           (6371 * acos(
              cos(radians(%s)) * cos(radians(gps_latitude)) * cos(radians(gps_longitude) - radians(%s)) +
              sin(radians(%s)) * sin(radians(gps_latitude))
           )) AS distance_km
    FROM images
    WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL
      AND (6371 * acos(
              cos(radians(%s)) * cos(radians(gps_latitude)) * cos(radians(gps_longitude) - radians(%s)) +
              sin(radians(%s)) * sin(radians(gps_latitude))
           )) <= %s
    ORDER BY distance_km ASC
    LIMIT %s
    """
    params = (lat, lon, lat, lat, lon, lat, radius_km, limit)
    try:
        with psycopg2.connect(conn_str) as local_conn:
            local_conn.autocommit = True
            with local_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return cur.fetchall()
    except Exception as e:
        msg = str(e)
        if 'relation "images" does not exist' in msg:
            st.error('Database schema not initialized (images table missing). Click "Initialize schema" in the sidebar to create it.')
            if st.sidebar.button('Initialize schema'):
                init_schema(conn_str)
        else:
            st.error(f"Location query failed: {e}")
        return []


def show_results_grid(rows, cols=3, thumb_width=300):
    if not rows:
        st.info("No results")
        return
    for i in range(0, len(rows), cols):
        cols_ui = st.columns(cols)
        for j, row in enumerate(rows[i:i+cols]):
            with cols_ui[j]:
                path = row.get('file_path') or row.get('path')
                try:
                    st.image(path, width=thumb_width)
                except Exception:
                    st.text("Could not load image")
                file_name = row.get('file_name') or os.path.basename(path)
                st.markdown(f"**{file_name}**")
                # Show YYYY/MM/DD directory path if available
                if path:
                    dir_path = os.path.dirname(path)
                    # Extract last 3 components for YYYY/MM/DD format
                    path_parts = dir_path.split(os.sep)
                    if len(path_parts) >= 3:
                        date_dir = os.sep.join(path_parts[-3:])
                        st.caption(f"📁 {date_dir}")
                    elif dir_path:
                        st.caption(f"📁 {dir_path}")
                cap = row.get('caption')
                if cap:
                    st.caption(cap)
                if row.get('gps_latitude') and row.get('gps_longitude'):
                    st.write(f"📍 {row['gps_latitude']:.5f}, {row['gps_longitude']:.5f}")


def main():
    st.title("Photo Archive Explorer")

    st.sidebar.header("Connection")
    conn_str = st.sidebar.text_input("Postgres connection string", value=os.environ.get('IMAGE_ARCHIVE_DB', ''))
    if st.sidebar.button("Connect"):
        conn = connect_db(conn_str)
        if conn:
            st.session_state['conn_str'] = conn_str
            st.session_state['conn'] = conn

    conn = st.session_state.get('conn') if st.session_state.get('conn_str') == conn_str else None
    if conn:
        stats = get_stats(conn)
        st.sidebar.metric("Total images", stats.get('total'))
        st.sidebar.metric("With captions", stats.get('with_captions'))
        st.sidebar.metric("With GPS", stats.get('with_gps'))

    st.sidebar.markdown("---")
    st.sidebar.header("Search")
    mode = st.sidebar.selectbox("Search mode", ["Text ILIKE", "pgvector (requires embeddings)"])
    if mode == 'pgvector (requires embeddings)':
        emb_provider = st.sidebar.selectbox("Embedding provider", ["local-sentence-transformers", "none"])
        model_name = st.sidebar.text_input("Local model (sentence-transformers)", value="all-MiniLM-L6-v2")

    q = st.sidebar.text_input("Caption query")
    limit = st.sidebar.slider("Results", min_value=1, max_value=200, value=24)

    st.header("Explore")
    if not conn:
        st.info("Connect to your PostgreSQL database first (provide connection string in sidebar)")
        return

    with st.expander("Query by caption"):
        if st.button("Run caption query"):
            if not q:
                st.warning("Enter a query")
            else:
                if mode == 'Text ILIKE':
                    rows = query_caption_text(q, limit=limit)
                    show_results_grid(rows)
                else:
                    if emb_provider == 'local-sentence-transformers':
                        vec = embed_query_local(model_name, q)
                        if vec:
                            rows = query_caption_pgvector(vec, limit=limit)
                            show_results_grid(rows)
                    else:
                        st.warning('No embedding provider selected')

    with st.expander("Query by GPS proximity"):
        gps_lat = st.number_input("Latitude", format="%.6f")
        gps_lon = st.number_input("Longitude", format="%.6f")
        radius = st.number_input("Radius (km)", value=5.0)
        if st.button("Find nearby"):
            if gps_lat == 0 and gps_lon == 0:
                st.warning("Enter latitude and longitude")
            else:
                rows = query_metadata_location(gps_lat, gps_lon, radius_km=radius, limit=limit)
                show_results_grid(rows)


if __name__ == '__main__':
    main()
