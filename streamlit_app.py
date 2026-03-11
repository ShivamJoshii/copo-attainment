"""
CO-PO Attainment System - Standalone Streamlit App
For deployment on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import re
import numpy as np
import os

# Sentence embeddings for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Cache the model at module level
_embedding_model = None

def get_embedding_model():
    """Lazy load the sentence transformer model"""
    global _embedding_model
    if _embedding_model is None and EMBEDDINGS_AVAILABLE:
        # Use a lightweight but effective model
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

# Page configuration
st.set_page_config(
    page_title="CO-PO Attainment System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ DATABASE SETUP ============
@st.cache_resource
def get_db_connection():
    """Create SQLite database connection"""
    conn = sqlite3.connect('copo_data.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Courses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            code TEXT NOT NULL,
            internal_weight REAL DEFAULT 0.4,
            ese_weight REAL DEFAULT 0.6,
            direct_weight REAL DEFAULT 0.8,
            indirect_weight REAL DEFAULT 0.2,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Course Outcomes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS course_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            co_code TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
        )
    ''')
    
    # Program Outcomes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS program_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            po_code TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
        )
    ''')
    
    # CO-PO Mappings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS co_po_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            co_id INTEGER NOT NULL,
            po_id INTEGER NOT NULL,
            weight INTEGER DEFAULT 0,
            similarity_score REAL,
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
            FOREIGN KEY (co_id) REFERENCES course_outcomes(id) ON DELETE CASCADE,
            FOREIGN KEY (po_id) REFERENCES program_outcomes(id) ON DELETE CASCADE,
            UNIQUE(course_id, co_id, po_id)
        )
    ''')
    
    # Attainment Inputs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attainment_inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            co_id INTEGER NOT NULL,
            internal_level INTEGER,
            ese_level INTEGER,
            indirect_value REAL,
            target_level REAL DEFAULT 1.4,
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
            FOREIGN KEY (co_id) REFERENCES course_outcomes(id) ON DELETE CASCADE,
            UNIQUE(course_id, co_id)
        )
    ''')
    
    # Calculated Results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS calculated_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            co_id INTEGER NOT NULL,
            direct_attainment REAL,
            final_attainment REAL,
            scale_of_3 REAL,
            target_level REAL,
            target_achieved TEXT,
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
            FOREIGN KEY (co_id) REFERENCES course_outcomes(id) ON DELETE CASCADE,
            UNIQUE(course_id, co_id)
        )
    ''')
    
    # PO Attainment Results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS po_attainment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL,
            po_id INTEGER NOT NULL,
            attainment_value REAL,
            attainment_percentage REAL,
            scale_of_3 REAL,
            target_achieved TEXT,
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
            FOREIGN KEY (po_id) REFERENCES program_outcomes(id) ON DELETE CASCADE,
            UNIQUE(course_id, po_id)
        )
    ''')
    
    conn.commit()

# Initialize database
init_db()

# ============ CALCULATION FUNCTIONS ============
def calculate_direct_attainment(internal_level: float, ese_level: float, 
                                 internal_weight: float = 0.4, ese_weight: float = 0.6) -> float:
    """Direct = internal_weight × Internal + ese_weight × ESE"""
    return internal_weight * internal_level + ese_weight * ese_level

def calculate_final_attainment(direct: float, indirect: float,
                               direct_weight: float = 0.8, indirect_weight: float = 0.2) -> float:
    """Final = direct_weight × Direct + indirect_weight × Indirect"""
    return direct_weight * direct + indirect_weight * indirect

def calculate_scale_of_3(final_attainment: float) -> float:
    """Scale of 3 = Final Attainment × 3"""
    return final_attainment * 3

def check_target_achieved(scale_of_3: float, target_level: float) -> str:
    """Returns 'Y' if scale >= target, else 'N'"""
    return "Y" if scale_of_3 >= target_level else "N"

def percentage_to_level(percentage: float) -> int:
    """Convert percentage to attainment level (0-3)"""
    if percentage >= 70:
        return 3
    elif percentage >= 60:
        return 2
    elif percentage >= 50:
        return 1
    else:
        return 0

# ============ NLP MAPPING (ENHANCED) ============

# Domain-specific keywords for engineering/business education
DOMAIN_KEYWORDS = {
    # Technical/Engineering
    'technical': ['technical', 'engineering', 'system', 'design', 'analysis', 'analyze', 'model', 
                  'modeling', 'simulation', 'algorithm', 'implementation', 'development', 'architecture'],
    
    # Problem Solving
    'problem_solving': ['problem', 'solve', 'solution', 'troubleshoot', 'debug', 'optimize', 
                        'improve', 'evaluate', 'assess', 'diagnose', 'resolve'],
    
    # Programming/Computing
    'computing': ['program', 'programming', 'code', 'coding', 'software', 'application', 
                  'database', 'data', 'computation', 'computing', 'automation', 'script'],
    
    # Communication
    'communication': ['communicate', 'communication', 'present', 'presentation', 'report', 
                      'document', 'documentation', 'write', 'writing', 'explain', 'describe'],
    
    # Teamwork/Collaboration
    'teamwork': ['team', 'collaborate', 'collaboration', 'group', 'cooperate', 'cooperation',
                 'coordinate', 'coordination', 'interpersonal', 'leadership', 'manage'],
    
    # Ethics/Professionalism
    'ethics': ['ethic', 'ethical', 'professional', 'professionalism', 'responsibility', 
               'responsible', 'integrity', 'safety', 'sustainability', 'sustainable'],
    
    # Learning/Research
    'learning': ['learn', 'learning', 'research', 'investigate', 'study', 'explore', 
                 'discover', 'innovation', 'innovate', 'adapt', 'lifelong'],
    
    # Business/Management
    'business': ['business', 'management', 'project', 'planning', 'strategy', 'strategic',
                 'finance', 'financial', 'economic', 'economics', 'marketing', 'entrepreneurship'],
    
    # Mathematics
    'math': ['mathematics', 'mathematical', 'math', 'calculation', 'calculate', 'formula',
             'equation', 'statistical', 'statistics', 'probability', 'numerical'],
    
    # Science
    'science': ['science', 'scientific', 'experiment', 'experimental', 'laboratory', 'lab',
                'empirical', 'hypothesis', 'theory', 'theoretical', 'physics', 'chemistry']
}

# Bloom's taxonomy verbs mapped to cognitive levels
BLOOMS_TAXONOMY = {
    'remember': ['define', 'identify', 'list', 'name', 'recall', 'recognize', 'state', 'describe'],
    'understand': ['explain', 'interpret', 'classify', 'summarize', 'paraphrase', 'compare', 'contrast'],
    'apply': ['apply', 'use', 'demonstrate', 'execute', 'implement', 'solve', 'calculate', 'compute'],
    'analyze': ['analyze', 'differentiate', 'organize', 'compare', 'contrast', 'distinguish', 'examine'],
    'evaluate': ['evaluate', 'assess', 'critique', 'judge', 'test', 'recommend', 'justify', 'validate'],
    'create': ['design', 'develop', 'create', 'formulate', 'construct', 'program', 'build', 'invent']
}

def preprocess_text(text: str) -> dict:
    """
    Enhanced text preprocessing with domain keyword extraction.
    Returns a dict with processed tokens, domain categories, and Bloom's levels.
    """
    # Clean and normalize
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    words = set(re.findall(r'\b[a-z]{3,}\b', text_clean))
    
    # Extract domain categories present in text
    domain_matches = {}
    for category, keywords in DOMAIN_KEYWORDS.items():
        matches = words & set(keywords)
        if matches:
            domain_matches[category] = len(matches)
    
    # Detect Bloom's taxonomy levels
    bloom_levels = set()
    for level, verbs in BLOOMS_TAXONOMY.items():
        if words & set(verbs):
            bloom_levels.add(level)
    
    # Extract key phrases (2-3 word combinations)
    text_words = re.findall(r'\b[a-z]+\b', text_clean)
    phrases = set()
    for i in range(len(text_words) - 1):
        phrases.add(f"{text_words[i]} {text_words[i+1]}")
        if i < len(text_words) - 2:
            phrases.add(f"{text_words[i]} {text_words[i+1]} {text_words[i+2]}")
    
    return {
        'words': words,
        'domain_matches': domain_matches,
        'bloom_levels': bloom_levels,
        'phrases': phrases,
        'raw_text': text_clean
    }

def calculate_semantic_similarity(co_processed: dict, po_processed: dict) -> float:
    """
    Calculate semantic similarity between CO and PO using multiple factors.
    """
    scores = []
    
    # 1. Word overlap (Jaccard) - baseline
    word_intersection = len(co_processed['words'] & po_processed['words'])
    word_union = len(co_processed['words'] | po_processed['words'])
    word_similarity = word_intersection / word_union if word_union > 0 else 0
    scores.append(word_similarity * 0.3)  # 30% weight
    
    # 2. Domain category overlap - strong indicator
    co_domains = set(co_processed['domain_matches'].keys())
    po_domains = set(po_processed['domain_matches'].keys())
    if co_domains and po_domains:
        domain_intersection = len(co_domains & po_domains)
        domain_union = len(co_domains | po_domains)
        domain_similarity = domain_intersection / domain_union if domain_union > 0 else 0
        
        # Boost score if multiple domains match
        if domain_intersection >= 2:
            domain_similarity = min(1.0, domain_similarity * 1.3)
        scores.append(domain_similarity * 0.4)  # 40% weight
    else:
        scores.append(0)
    
    # 3. Bloom's taxonomy alignment
    co_blooms = co_processed['bloom_levels']
    po_blooms = po_processed['bloom_levels']
    if co_blooms and po_blooms:
        bloom_overlap = len(co_blooms & po_blooms)
        bloom_similarity = bloom_overlap / max(len(co_blooms), len(po_blooms)) if max(len(co_blooms), len(po_blooms)) > 0 else 0
        scores.append(bloom_similarity * 0.2)  # 20% weight
    else:
        scores.append(0)
    
    # 4. Phrase overlap - captures semantic meaning better
    phrase_intersection = len(co_processed['phrases'] & po_processed['phrases'])
    phrase_union = len(co_processed['phrases'] | po_processed['phrases'])
    phrase_similarity = phrase_intersection / phrase_union if phrase_union > 0 else 0
    scores.append(phrase_similarity * 0.1)  # 10% weight
    
    return sum(scores)

def generate_co_po_mapping_simple(co_descriptions: List[str], po_descriptions: List[str]) -> List[List[int]]:
    """
    Enhanced CO-PO mapping using sentence embeddings + cosine similarity.
    Falls back to rule-based approach if embeddings not available.
    """
    matrix = []
    
    # Try sentence embeddings first
    if EMBEDDINGS_AVAILABLE and len(co_descriptions) > 0 and len(po_descriptions) > 0:
        try:
            model = get_embedding_model()
            if model:
                # Generate embeddings for all descriptions
                co_embeddings = model.encode(co_descriptions, convert_to_numpy=True)
                po_embeddings = model.encode(po_descriptions, convert_to_numpy=True)
                
                # Calculate cosine similarity matrix
                similarity_matrix = cosine_similarity(co_embeddings, po_embeddings)
                
                # Convert similarities to weights (0-3)
                # Optimized thresholds based on expert benchmark analysis
                # 0→1: 0.265, 1→2: 0.397, 2→3: 0.518
                for i in range(len(co_descriptions)):
                    row = []
                    for j in range(len(po_descriptions)):
                        sim = similarity_matrix[i][j]
                        
                        # Optimized thresholds for best alignment with expert ratings
                        if sim < 0.265:
                            weight = 0
                        elif sim < 0.397:
                            weight = 1
                        elif sim < 0.518:
                            weight = 2
                        else:
                            weight = 3
                        
                        row.append(weight)
                    matrix.append(row)
                
                return matrix
        except Exception as e:
            st.warning(f"Embedding-based mapping failed, falling back to rule-based: {e}")
    
    # Fallback to rule-based approach
    co_processed = [preprocess_text(desc) for desc in co_descriptions]
    po_processed = [preprocess_text(desc) for desc in po_descriptions]
    
    for co_data in co_processed:
        row = []
        for po_data in po_processed:
            similarity = calculate_semantic_similarity(co_data, po_data)
            
            # Optimized thresholds for fallback too
            if similarity < 0.265:
                weight = 0
            elif similarity < 0.397:
                weight = 1
            elif similarity < 0.518:
                weight = 2
            else:
                weight = 3
            
            row.append(weight)
        matrix.append(row)
    
    return matrix

# ============ DATABASE OPERATIONS ============
def get_courses():
    """Get all courses"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM courses ORDER BY created_at DESC")
    return [dict(row) for row in cursor.fetchall()]

def create_course(name: str, code: str):
    """Create a new course"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO courses (name, code) VALUES (?, ?)",
        (name, code)
    )
    conn.commit()
    return cursor.lastrowid

def delete_course(course_id: int):
    """Delete a course and all related data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM courses WHERE id = ?", (course_id,))
    conn.commit()

def get_course_outcomes(course_id: int):
    """Get COs for a course"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM course_outcomes WHERE course_id = ? ORDER BY co_code", (course_id,))
    return [dict(row) for row in cursor.fetchall()]

def create_course_outcomes(course_id: int, outcomes: List[Dict]):
    """Create COs for a course"""
    conn = get_db_connection()
    cursor = conn.cursor()
    for outcome in outcomes:
        cursor.execute(
            "INSERT INTO course_outcomes (course_id, co_code, description) VALUES (?, ?, ?)",
            (course_id, outcome['co_code'], outcome['description'])
        )
    conn.commit()

def get_program_outcomes(course_id: int):
    """Get POs for a course"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM program_outcomes WHERE course_id = ? ORDER BY po_code", (course_id,))
    return [dict(row) for row in cursor.fetchall()]

def create_program_outcomes(course_id: int, outcomes: List[Dict]):
    """Create POs for a course"""
    conn = get_db_connection()
    cursor = conn.cursor()
    for outcome in outcomes:
        cursor.execute(
            "INSERT INTO program_outcomes (course_id, po_code, description) VALUES (?, ?, ?)",
            (course_id, outcome['po_code'], outcome['description'])
        )
    conn.commit()

def get_co_po_mappings(course_id: int):
    """Get CO-PO mappings for a course"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT m.*, co.co_code, po.po_code 
        FROM co_po_mappings m
        JOIN course_outcomes co ON m.co_id = co.id
        JOIN program_outcomes po ON m.po_id = po.id
        WHERE m.course_id = ?
    ''', (course_id,))
    return [dict(row) for row in cursor.fetchall()]

def create_co_po_mappings(course_id: int, mappings: List[Dict]):
    """Create CO-PO mappings"""
    conn = get_db_connection()
    cursor = conn.cursor()
    for mapping in mappings:
        cursor.execute('''
            INSERT OR REPLACE INTO co_po_mappings 
            (course_id, co_id, po_id, weight, similarity_score) 
            VALUES (?, ?, ?, ?, ?)
        ''', (course_id, mapping['co_id'], mapping['po_id'], mapping['weight'], mapping.get('similarity_score', 0)))
    conn.commit()

def generate_and_save_mapping(course_id: int):
    """Generate CO-PO mapping using NLP and save to database"""
    cos = get_course_outcomes(course_id)
    pos = get_program_outcomes(course_id)
    
    if not cos or not pos:
        return []
    
    co_descs = [co['description'] for co in cos]
    po_descs = [po['description'] for po in pos]
    
    # Generate mapping matrix
    matrix = generate_co_po_mapping_simple(co_descs, po_descs)
    
    # Save mappings
    mappings = []
    for i, co in enumerate(cos):
        for j, po in enumerate(pos):
            mappings.append({
                'co_id': co['id'],
                'po_id': po['id'],
                'weight': matrix[i][j],
                'similarity_score': matrix[i][j] / 3.0
            })
    
    create_co_po_mappings(course_id, mappings)
    return mappings

def get_attainment_inputs(course_id: int):
    """Get attainment inputs for a course"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT i.*, co.co_code 
        FROM attainment_inputs i
        JOIN course_outcomes co ON i.co_id = co.id
        WHERE i.course_id = ?
    ''', (course_id,))
    return [dict(row) for row in cursor.fetchall()]

def save_attainment_inputs(course_id: int, inputs: List[Dict]):
    """Save attainment inputs"""
    conn = get_db_connection()
    cursor = conn.cursor()
    for inp in inputs:
        cursor.execute('''
            INSERT OR REPLACE INTO attainment_inputs 
            (course_id, co_id, internal_level, ese_level, indirect_value, target_level)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (course_id, inp['co_id'], inp['internal_level'], inp['ese_level'], 
              inp['indirect_value'], inp.get('target_level', 1.4)))
    conn.commit()

def calculate_and_save_results(course_id: int, target_level: float = 1.4):
    """Calculate CO and PO attainment results"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get course weights
    cursor.execute("SELECT * FROM courses WHERE id = ?", (course_id,))
    course = dict(cursor.fetchone())
    
    # Get inputs
    inputs = get_attainment_inputs(course_id)
    cos = get_course_outcomes(course_id)
    pos = get_program_outcomes(course_id)
    mappings = get_co_po_mappings(course_id)
    
    if not inputs:
        return None
    
    # Calculate CO results
    co_results = {}
    for inp in inputs:
        direct = calculate_direct_attainment(
            inp['internal_level'], inp['ese_level'],
            course['internal_weight'], course['ese_weight']
        )
        final = calculate_final_attainment(
            direct, inp['indirect_value'],
            course['direct_weight'], course['indirect_weight']
        )
        scale = calculate_scale_of_3(final)
        achieved = check_target_achieved(scale, target_level)
        
        co_results[inp['co_id']] = {
            'direct_attainment': direct,
            'final_attainment': final,
            'scale_of_3': scale,
            'target_achieved': achieved
        }
        
        # Save to database
        cursor.execute('''
            INSERT OR REPLACE INTO calculated_results 
            (course_id, co_id, direct_attainment, final_attainment, scale_of_3, target_level, target_achieved)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (course_id, inp['co_id'], direct, final, scale, target_level, achieved))
    
    # Build mapping matrix
    mapping_matrix = {}
    for m in mappings:
        if m['co_id'] not in mapping_matrix:
            mapping_matrix[m['co_id']] = {}
        mapping_matrix[m['co_id']][m['po_id']] = m['weight']
    
    # Calculate weighted PO attainments
    for po in pos:
        po_id = po['id']
        weighted_sum = 0.0
        total_weight = 0
        
        for co in cos:
            co_id = co['id']
            if co_id in co_results and co_id in mapping_matrix and po_id in mapping_matrix[co_id]:
                weight = mapping_matrix[co_id][po_id]
                if weight > 0:
                    weighted_sum += co_results[co_id]['final_attainment'] * weight
                    total_weight += weight
        
        if total_weight > 0:
            attainment_value = weighted_sum / total_weight
            attainment_percentage = (attainment_value / 3) * 100
            scale_of_3 = calculate_scale_of_3(attainment_value)
            target_achieved = check_target_achieved(scale_of_3, target_level)
        else:
            attainment_value = 0.0
            attainment_percentage = 0.0
            scale_of_3 = 0.0
            target_achieved = "N"
        
        # Save to database
        cursor.execute('''
            INSERT OR REPLACE INTO po_attainment_results 
            (course_id, po_id, attainment_value, attainment_percentage, scale_of_3, target_achieved)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (course_id, po_id, attainment_value, attainment_percentage, scale_of_3, target_achieved))
    
    conn.commit()
    return {'co_results': co_results}

def get_calculated_results(course_id: int):
    """Get calculated CO results"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT r.*, co.co_code 
        FROM calculated_results r
        JOIN course_outcomes co ON r.co_id = co.id
        WHERE r.course_id = ?
    ''', (course_id,))
    return [dict(row) for row in cursor.fetchall()]

def get_po_results(course_id: int):
    """Get calculated PO results"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT r.*, po.po_code 
        FROM po_attainment_results r
        JOIN program_outcomes po ON r.po_id = po.id
        WHERE r.course_id = ?
    ''', (course_id,))
    return [dict(row) for row in cursor.fetchall()]

# ============ UI COMPONENTS ============
def sidebar_navigation():
    """Render sidebar navigation"""
    st.sidebar.title("📊 CO-PO Attainment")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Dashboard": "dashboard",
        "➕ Create Course": "create",
        "📋 Enter Attainments": "attainments",
        "📈 View Results": "results"
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("CO-PO Attainment System for NBA compliance. Automates mapping COs to POs and calculates attainment from student assessments.")
    
    return pages[selection]

def render_dashboard():
    """Render dashboard page"""
    st.title("🏠 Dashboard")
    
    courses = get_courses()
    
    if not courses:
        st.info("No courses yet. Create your first course to get started!")
        if st.button("➕ Create Course", type="primary"):
            st.session_state.page = "create"
            st.rerun()
        return
    
    st.subheader("All Courses")
    
    for course in courses:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 2])
            
            with col1:
                st.write(f"**{course['code']}** - {course['name']}")
            
            with col2:
                results = get_calculated_results(course['id'])
                if results:
                    achieved = sum(1 for r in results if r['target_achieved'] == 'Y')
                    st.write(f"✅ {achieved}/{len(results)} COs meeting target")
                else:
                    st.write("⏳ No calculations yet")
            
            with col3:
                if st.button("📋 Enter Attainments", key=f"att_{course['id']}"):
                    st.session_state.selected_course = course['id']
                    st.session_state.page = "attainments"
                    st.rerun()
                if st.button("📈 View Results", key=f"res_{course['id']}"):
                    st.session_state.selected_course = course['id']
                    st.session_state.page = "results"
                    st.rerun()
            
            st.divider()

def render_create_course():
    """Render create course wizard"""
    st.title("➕ Create Course")
    
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
        st.session_state.new_course_id = None
    
    step = st.session_state.wizard_step
    progress = (step - 1) / 4
    st.progress(progress, text=f"Step {step} of 5")
    
    if step == 1:
        st.subheader("Step 1: Course Information")
        
        with st.form("course_info"):
            course_name = st.text_input("Course Name", placeholder="e.g., Compiler Design")
            course_code = st.text_input("Course Code", placeholder="e.g., CS6A")
            
            submitted = st.form_submit_button("Next", type="primary")
            
            if submitted:
                if course_name and course_code:
                    course_id = create_course(course_name, course_code)
                    st.session_state.new_course_id = course_id
                    st.session_state.wizard_step = 2
                    st.rerun()
                else:
                    st.error("Please fill in all fields")
    
    elif step == 2:
        st.subheader("Step 2: Course Outcomes (COs)")
        st.write("Enter 3-6 Course Outcomes describing what students will learn.")
        
        with st.form("cos_form"):
            cos_data = []
            for i in range(1, 7):
                co_code = f"CO{i}"
                description = st.text_area(f"{co_code}", placeholder=f"Description for {co_code}", key=f"co_{i}")
                if description.strip():
                    cos_data.append({'co_code': co_code, 'description': description})
            
            submitted = st.form_submit_button("Next", type="primary")
            
            if submitted:
                if len(cos_data) >= 3:
                    create_course_outcomes(st.session_state.new_course_id, cos_data)
                    st.session_state.wizard_step = 3
                    st.rerun()
                else:
                    st.error("Please enter at least 3 Course Outcomes")
    
    elif step == 3:
        st.subheader("Step 3: Program Outcomes (POs)")
        
        default_pos = [
            ("PO1", "Engineering knowledge: Apply knowledge of mathematics, science, and engineering"),
            ("PO2", "Problem analysis: Identify, formulate, and solve engineering problems"),
            ("PO3", "Design/development of solutions: Design solutions for complex engineering problems"),
            ("PO4", "Conduct investigations of complex problems"),
            ("PO5", "Modern tool usage"),
            ("PO6", "The engineer and society"),
            ("PO7", "Environment and sustainability"),
            ("PO8", "Ethics"),
            ("PO9", "Individual and team work"),
            ("PO10", "Communication"),
            ("PO11", "Project management and finance"),
            ("PO12", "Life-long learning")
        ]
        
        with st.form("pos_form"):
            pos_data = []
            for po_code, default_desc in default_pos:
                use_po = st.checkbox(f"Include {po_code}", value=True, key=f"chk_{po_code}")
                if use_po:
                    description = st.text_area(f"{po_code}", value=default_desc, key=f"po_{po_code}")
                    pos_data.append({'po_code': po_code, 'description': description})
            
            submitted = st.form_submit_button("Next", type="primary")
            
            if submitted:
                if len(pos_data) >= 3:
                    create_program_outcomes(st.session_state.new_course_id, pos_data)
                    st.session_state.wizard_step = 4
                    st.rerun()
                else:
                    st.error("Please select at least 3 Program Outcomes")
    
    elif step == 4:
        st.subheader("Step 4: CO-PO Mapping")
        st.write("Review and adjust the auto-generated CO-PO mapping.")
        
        mappings = get_co_po_mappings(st.session_state.new_course_id)
        if not mappings:
            mappings = generate_and_save_mapping(st.session_state.new_course_id)
            st.rerun()
        
        cos = get_course_outcomes(st.session_state.new_course_id)
        pos = get_program_outcomes(st.session_state.new_course_id)
        
        matrix_data = {}
        for m in mappings:
            key = (m['co_code'], m['po_code'])
            matrix_data[key] = m['weight']
        
        # Display header row first
        header_cols = st.columns(len(pos) + 1)
        header_cols[0].write("**CO \\ PO**")
        for j, po in enumerate(pos):
            header_cols[j + 1].write(f"**{po['po_code']}**")
        
        st.divider()
        
        edited_weights = {}
        for co in cos:
            cols = st.columns(len(pos) + 1)
            cols[0].write(f"**{co['co_code']}**")
            for j, po in enumerate(pos):
                key = (co['co_code'], po['po_code'])
                current_weight = matrix_data.get(key, 0)
                edited_weights[key] = cols[j + 1].number_input(
                    f"{co['co_code']}-{po['po_code']}",
                    min_value=0, max_value=3, value=current_weight,
                    label_visibility="collapsed",
                    key=f"map_{co['id']}_{po['id']}"
                )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.wizard_step = 3
                st.rerun()
        with col2:
            if st.button("Save & Continue", type="primary"):
                co_dict = {co['co_code']: co['id'] for co in cos}
                po_dict = {po['po_code']: po['id'] for po in pos}
                
                updated_mappings = []
                for (co_code, po_code), weight in edited_weights.items():
                    updated_mappings.append({
                        'co_id': co_dict[co_code],
                        'po_id': po_dict[po_code],
                        'weight': weight,
                        'similarity_score': weight / 3.0
                    })
                
                create_co_po_mappings(st.session_state.new_course_id, updated_mappings)
                st.session_state.wizard_step = 5
                st.rerun()
    
    elif step == 5:
        st.subheader("Step 5: Done!")
        st.success("Course created successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Enter Attainments"):
                st.session_state.selected_course = st.session_state.new_course_id
                st.session_state.wizard_step = 1
                st.session_state.page = "attainments"
                st.rerun()
        with col2:
            if st.button("🏠 Go to Dashboard"):
                st.session_state.wizard_step = 1
                st.session_state.page = "dashboard"
                st.rerun()

def render_attainments():
    """Render attainments input page"""
    st.title("📋 Enter Attainments")
    
    courses = get_courses()
    if not courses:
        st.warning("No courses available. Create a course first.")
        return
    
    course_options = {f"{c['code']} - {c['name']}": c['id'] for c in courses}
    selected = st.selectbox("Select Course", list(course_options.keys()))
    course_id = course_options[selected]
    st.session_state.selected_course = course_id
    
    cos = get_course_outcomes(course_id)
    if not cos:
        st.warning("No Course Outcomes found for this course.")
        return
    
    existing_inputs = {inp['co_id']: inp for inp in get_attainment_inputs(course_id)}
    
    st.subheader(f"Attainment Inputs for {selected}")
    
    with st.form("attainment_form"):
        target_level = st.number_input("Target Level", min_value=0.0, max_value=3.0, value=1.4, step=0.1)
        
        inputs_data = []
        for co in cos:
            st.markdown(f"**{co['co_code']}**: {co['description'][:100]}...")
            
            existing = existing_inputs.get(co['id'], {})
            
            cols = st.columns(3)
            with cols[0]:
                internal = st.number_input(
                    f"Internal ({co['co_code']})", min_value=0, max_value=3,
                    value=existing.get('internal_level', 0), key=f"int_{co['id']}"
                )
            with cols[1]:
                ese = st.number_input(
                    f"ESE ({co['co_code']})", min_value=0, max_value=3,
                    value=existing.get('ese_level', 0), key=f"ese_{co['id']}"
                )
            with cols[2]:
                indirect = st.number_input(
                    f"Indirect ({co['co_code']})", min_value=0.0, max_value=3.0,
                    value=existing.get('indirect_value', 0.0), step=0.1, key=f"ind_{co['id']}"
                )
            
            inputs_data.append({
                'co_id': co['id'],
                'internal_level': internal,
                'ese_level': ese,
                'indirect_value': indirect,
                'target_level': target_level
            })
            
            st.divider()
        
        submitted = st.form_submit_button("💾 Save & Calculate", type="primary")
        
        if submitted:
            save_attainment_inputs(course_id, inputs_data)
            calculate_and_save_results(course_id, target_level)
            st.success("Attainment data saved and calculations completed!")
            st.balloons()

def render_results():
    """Render results page"""
    st.title("📈 View Results")
    
    courses = get_courses()
    if not courses:
        st.warning("No courses available.")
        return
    
    course_options = {f"{c['code']} - {c['name']}": c['id'] for c in courses}
    selected = st.selectbox("Select Course", list(course_options.keys()))
    course_id = course_options[selected]
    st.session_state.selected_course = course_id
    
    st.subheader(f"Results for {selected}")
    
    co_results = get_calculated_results(course_id)
    if co_results:
        st.subheader("CO Attainment Results")
        
        df_co = pd.DataFrame(co_results)
        df_co = df_co[['co_code', 'direct_attainment', 'final_attainment', 'scale_of_3', 'target_level', 'target_achieved']]
        df_co.columns = ['CO', 'Direct', 'Final', 'Scale of 3', 'Target', 'Achieved']
        
        st.dataframe(df_co, use_container_width=True)
        
        achieved_count = sum(1 for r in co_results if r['target_achieved'] == 'Y')
        total_count = len(co_results)
        avg_scale = sum(r['scale_of_3'] for r in co_results) / total_count if total_count > 0 else 0
        
        cols = st.columns(3)
        cols[0].metric("COs Meeting Target", f"{achieved_count}/{total_count}")
        cols[1].metric("Average Scale of 3", f"{avg_scale:.2f}")
        cols[2].metric("Success Rate", f"{(achieved_count/total_count*100):.1f}%")
    else:
        st.info("No calculations yet. Enter attainment data first.")
    
    po_results = get_po_results(course_id)
    if po_results:
        st.subheader("PO Attainment Results")
        
        df_po = pd.DataFrame(po_results)
        df_po = df_po[['po_code', 'attainment_value', 'attainment_percentage', 'scale_of_3', 'target_achieved']]
        df_po.columns = ['PO', 'Attainment', 'Percentage', 'Scale of 3', 'Achieved']
        
        st.dataframe(df_po, use_container_width=True)
        
        avg_po = sum(r['attainment_value'] for r in po_results) / len(po_results) if po_results else 0
        st.metric("Average PO Attainment", f"{avg_po:.2f}")

# ============ MAIN ============
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "dashboard"
    if 'selected_course' not in st.session_state:
        st.session_state.selected_course = None
    
    # Sidebar navigation
    page = sidebar_navigation()
    
    # Override with session state if set
    if st.session_state.page:
        page = st.session_state.page
    
    # Render selected page
    if page == "dashboard":
        render_dashboard()
    elif page == "create":
        render_create_course()
    elif page == "attainments":
        render_attainments()
    elif page == "results":
        render_results()

if __name__ == "__main__":
    main()
