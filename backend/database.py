# database.py
import mysql.connector
from mysql.connector import errorcode
import os
import json
import time
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

# --- Database Connection ---

@contextmanager
def get_db_connection():
    """Context manager for MySQL database connections"""
    conn = None
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )
        yield conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()


def init_database():
    """Initialize the database with required tables for MySQL"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        db_name = os.getenv("DB_NAME")
        print(f"Initializing tables in database '{db_name}'...")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scripts (
                script_id VARCHAR(255) PRIMARY KEY,
                topic TEXT NOT NULL,
                raw_script LONGTEXT NOT NULL,
                parsed_script JSON NOT NULL,
                created_at DOUBLE NOT NULL,
                updated_at DOUBLE NOT NULL
            ) ENGINE=InnoDB;
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS presentations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                script_id VARCHAR(255) NOT NULL,
                pdf_path VARCHAR(255) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                created_at DOUBLE NOT NULL,
                file_size BIGINT,
                pdf_images_path VARCHAR(255),
                FOREIGN KEY (script_id) REFERENCES scripts(script_id) ON DELETE CASCADE,
                UNIQUE(script_id)
            ) ENGINE=InnoDB;
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                script_id VARCHAR(255) NOT NULL,
                segment_idx INT NOT NULL,
                slide_idx INT NOT NULL,
                slide_key VARCHAR(255) NOT NULL,
                segment_title TEXT,
                segment_summary TEXT,
                slide_title TEXT,
                slide_narration TEXT,
                image_prompt TEXT,
                image_path VARCHAR(255),
                unsplash_url TEXT,
                source VARCHAR(50),
                created_at DOUBLE NOT NULL,
                FOREIGN KEY (script_id) REFERENCES scripts(script_id) ON DELETE CASCADE,
                UNIQUE(script_id, slide_key)
            ) ENGINE=InnoDB;
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audio (
                id INT AUTO_INCREMENT PRIMARY KEY,
                script_id VARCHAR(255) NOT NULL,
                audio_type VARCHAR(50) NOT NULL,
                segment_idx INT,
                slide_idx INT,
                content TEXT NOT NULL,
                audio_path VARCHAR(255) NOT NULL,
                speaker VARCHAR(50) NOT NULL,
                created_at DOUBLE NOT NULL,
                FOREIGN KEY (script_id) REFERENCES scripts(script_id) ON DELETE CASCADE,
                UNIQUE(script_id, audio_type, segment_idx, slide_idx)
            ) ENGINE=InnoDB;
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pdf_images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                script_id VARCHAR(255) NOT NULL,
                slide_number INT NOT NULL,
                image_path VARCHAR(255) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                file_size BIGINT NOT NULL,
                created_at DOUBLE NOT NULL,
                FOREIGN KEY (script_id) REFERENCES scripts(script_id) ON DELETE CASCADE,
                UNIQUE(script_id, slide_number)
            ) ENGINE=InnoDB;
        """
        )

        conn.commit()
        cursor.close()
        print("Database tables are ready.")


# --- CRUD Functions ---


def save_script_to_db(
    script_id: str, topic: str, raw_script: str, parsed_script: List[dict]
) -> bool:
    """Save or update script data in the MySQL database."""
    sql = """
        INSERT INTO scripts (script_id, topic, raw_script, parsed_script, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            topic = VALUES(topic),
            raw_script = VALUES(raw_script),
            parsed_script = VALUES(parsed_script),
            updated_at = VALUES(updated_at)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            current_time = time.time()
            cursor.execute(
                sql,
                (
                    script_id,
                    topic,
                    raw_script,
                    json.dumps(parsed_script),
                    current_time,
                    current_time,
                ),
            )
            conn.commit()
            cursor.close()
            return True
    except Exception as e:
        print(f"Error saving script to database: {e}")
        return False


def get_script_from_db(script_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve script data from the MySQL database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM scripts WHERE script_id = %s", (script_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row and isinstance(row.get("parsed_script"), str):
                row["parsed_script"] = json.loads(row["parsed_script"])
            return row
    except Exception as e:
        print(f"Error retrieving script from database: {e}")
        return None


def get_all_scripts_from_db() -> List[Dict[str, Any]]:
    """Retrieve all scripts metadata from the MySQL database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT script_id, topic, created_at, updated_at FROM scripts ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()
            cursor.close()
            return rows
    except Exception as e:
        print(f"Error retrieving scripts from database: {e}")
        return []


def save_images_to_db(script_id: str, images_data: Dict[str, Any]) -> bool:
    """Save image data to the MySQL database."""
    sql = """
        INSERT INTO images (script_id, segment_idx, slide_idx, slide_key, segment_title, segment_summary,
                            slide_title, slide_narration, image_prompt, image_path, unsplash_url, source, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            current_time = time.time()
            cursor.execute("DELETE FROM images WHERE script_id = %s", (script_id,))

            for slide_key, image_info in images_data.items():
                parts = slide_key.split("_")
                segment_idx = int(parts[1]) if len(parts) > 1 else 0
                slide_idx = int(parts[3]) if len(parts) > 3 else 0
                cursor.execute(
                    sql,
                    (
                        script_id,
                        segment_idx,
                        slide_idx,
                        slide_key,
                        image_info.get("segment_title"),
                        image_info.get("segment_summary"),
                        image_info.get("slide_title"),
                        image_info.get("slide_narration"),
                        image_info.get("image_prompt"),
                        image_info.get("image_path"),
                        image_info.get("unsplash_url"),
                        image_info.get("source"),
                        current_time,
                    ),
                )
            conn.commit()
            cursor.close()
            return True
    except Exception as e:
        print(f"Error saving images to database: {e}")
        return False


def get_images_from_db(script_id: str) -> Dict[str, Any]:
    """Retrieve image data from the MySQL database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM images WHERE script_id = %s ORDER BY segment_idx, slide_idx",
                (script_id,),
            )
            rows = cursor.fetchall()
            cursor.close()
            return {row["slide_key"]: row for row in rows}
    except Exception as e:
        print(f"Error retrieving images from database: {e}")
        return {}


def save_audio_to_db(script_id: str, audio_files: Dict[str, Any]) -> bool:
    """Save audio file paths to the MySQL database."""
    sql = """
        INSERT INTO audio (script_id, audio_type, segment_idx, slide_idx, content, audio_path, speaker, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            current_time = time.time()
            cursor.execute("DELETE FROM audio WHERE script_id = %s", (script_id,))

            for audio_info in audio_files.values():
                cursor.execute(
                    sql,
                    (
                        script_id,
                        audio_info.get("audio_type"),
                        audio_info.get("segment_idx"),
                        audio_info.get("slide_idx"),
                        audio_info.get("content"),
                        audio_info.get("audio_path"),
                        audio_info.get("speaker"),
                        current_time,
                    ),
                )
            conn.commit()
            cursor.close()
            return True
    except Exception as e:
        print(f"Error saving audio to database: {e}")
        return False


def get_audio_from_db(script_id: str) -> Dict[str, Any]:
    """Retrieve audio data from the MySQL database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM audio WHERE script_id = %s ORDER BY segment_idx, slide_idx",
                (script_id,),
            )
            rows = cursor.fetchall()
            cursor.close()
            audio_files = {}
            for row in rows:
                key = f"{row['audio_type']}_seg{row['segment_idx']}"
                if row["slide_idx"] is not None:
                    key += f"_slide{row['slide_idx']}"
                audio_files[key] = row
            return audio_files
    except Exception as e:
        print(f"Error retrieving audio from database: {e}")
        return {}


def save_presentation_to_db(
    script_id: str, pdf_path: str, filename: str, file_size: int
) -> bool:
    """Save presentation PDF path to the MySQL database."""
    sql = """
        INSERT INTO presentations (script_id, pdf_path, filename, created_at, file_size)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            pdf_path = VALUES(pdf_path),
            filename = VALUES(filename),
            file_size = VALUES(file_size)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                sql, (script_id, pdf_path, filename, time.time(), file_size)
            )
            conn.commit()
            cursor.close()
            return True
    except Exception as e:
        print(f"Error saving presentation to database: {e}")
        return False


def get_presentation_from_db(script_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve presentation data from the MySQL database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM presentations WHERE script_id = %s", (script_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            return row
    except Exception as e:
        print(f"Error retrieving presentation from database: {e}")
        return None


def save_pdf_images_to_db(script_id: str, images_data: List[dict]) -> bool:
    """Save individual PDF image paths to the MySQL database."""
    sql = """
        INSERT INTO pdf_images (script_id, slide_number, image_path, filename, file_size, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            current_time = time.time()
            cursor.execute("DELETE FROM pdf_images WHERE script_id = %s", (script_id,))
            for image_info in images_data:
                cursor.execute(
                    sql,
                    (
                        script_id,
                        image_info["slide_number"],
                        image_info["image_path"],
                        image_info["filename"],
                        image_info["file_size"],
                        current_time,
                    ),
                )
            conn.commit()
            cursor.close()
            return True
    except Exception as e:
        print(f"Error saving PDF images to database: {e}")
        return False


def get_pdf_images_from_db(script_id: str) -> List[Dict[str, Any]]:
    """Retrieve PDF image data from the MySQL database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM pdf_images WHERE script_id = %s ORDER BY slide_number",
                (script_id,),
            )
            rows = cursor.fetchall()
            cursor.close()
            return rows
    except Exception as e:
        print(f"Error retrieving PDF images from database: {e}")
        return []


def get_all_assets_from_db() -> List[Dict[str, Any]]:
    """Retrieve all scripts and their associated presentation PDFs from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT
                    s.script_id,
                    s.topic,
                    s.created_at as script_created_at,
                    p.pdf_path,
                    p.filename as pdf_filename,
                    p.file_size as pdf_file_size,
                    p.created_at as pdf_created_at
                FROM scripts s
                LEFT JOIN presentations p ON s.script_id = p.script_id
                ORDER BY s.created_at DESC
            """
            )
            rows = cursor.fetchall()
            cursor.close()
            return rows
    except Exception as e:
        print(f"Error retrieving all assets from database: {e}")
        return []