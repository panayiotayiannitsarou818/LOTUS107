# -*- coding: utf-8 -*-
"""
ΛΕΙΤΟΥΡΓΙΚΗ ΕΚΔΟΣΗ - Με βήματα ανάθεσης αλλά χωρίς περίπλοκα γραφήματα
"""

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
from typing import Dict, List, Tuple, Any
import traceback

# Import των modules που χρειάζονται
try:
    from statistics_generator import generate_statistics_table, export_statistics_to_excel
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    st.warning("⚠️ Module statistics_generator δεν βρέθηκε. Τα στατιστικά θα είναι περιορισμένα.")

# Streamlit configuration
st.set_page_config(
    page_title="Σύστημα Ανάθεσης Μαθητών",
    page_icon="🎓",
    layout="wide"
)

def init_session_state():
    """Αρχικοποίηση session state"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'results' not in st.session_state:
        st.session_state.results = {}

def safe_load_data(uploaded_file):
    """Ασφαλής φόρτωση και κανονικοποίηση δεδομένων"""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            return None, "Μη υποστηριζόμενο format αρχείου"
        
        # Κανονικοποίηση στηλών
        rename_map = {}
        for col in df.columns:
            col_str = str(col).strip().upper()
            if any(x in col_str for x in ['ΟΝΟΜΑ', 'NAME', 'ΜΑΘΗΤΗΣ']):
                rename_map[col] = 'ΟΝΟΜΑ'
            elif any(x in col_str for x in ['ΦΥΛΟ', 'GENDER']):
                rename_map[col] = 'ΦΥΛΟ'
            elif 'ΓΝΩΣΗ' in col_str and 'ΕΛΛΗΝΙΚ' in col_str:
                rename_map[col] = 'ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ'
            elif 'ΠΑΙΔΙ' in col_str and 'ΕΚΠΑΙΔΕΥΤΙΚ' in col_str:
                rename_map[col] = 'ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ'
            elif 'ΦΙΛΟΙ' in col_str or 'FRIEND' in col_str:
                rename_map[col] = 'ΦΙΛΟΙ'
            elif 'ΖΩΗΡ' in col_str:
                rename_map[col] = 'ΖΩΗΡΟΣ'
            elif 'ΙΔΙΑΙΤΕΡΟΤΗΤ' in col_str:
                rename_map[col] = 'ΙΔΙΑΙΤΕΡΟΤΗΤΑ'
            elif 'ΣΥΓΚΡΟΥΣ' in col_str:
                rename_map[col] = 'ΣΥΓΚΡΟΥΣΗ'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Κανονικοποίηση τιμών
        if 'ΦΥΛΟ' in df.columns:
            df['ΦΥΛΟ'] = df['ΦΥΛΟ'].astype(str).str.upper().map({
                'Α':'Α', 'Κ':'Κ', 'ΑΓΟΡΙ':'Α', 'ΚΟΡΙΤΣΙ':'Κ', 'BOY':'Α', 'GIRL':'Κ'
            }).fillna('Α')
        
        for col in ['ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ', 'ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ', 'ΖΩΗΡΟΣ', 'ΙΔΙΑΙΤΕΡΟΤΗΤΑ']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().map({
                    'ΝΑΙ':'Ν', 'ΟΧΙ':'Ο', 'YES':'Ν', 'NO':'Ο', '1':'Ν', '0':'Ο', 'TRUE':'Ν', 'FALSE':'Ο'
                }).fillna('Ο')
        
        return df, None
    except Exception as e:
        return None, f"Σφάλμα φόρτωσης: {str(e)}"

def display_basic_info(df):
    """Βασικές πληροφορίες"""
    st.subheader("📊 Βασικές Πληροφορίες")
    
    total_students = len(df)
    boys_count = len(df[df['ΦΥΛΟ'] == 'Α']) if 'ΦΥΛΟ' in df.columns else 0
    girls_count = len(df[df['ΦΥΛΟ'] == 'Κ']) if 'ΦΥΛΟ' in df.columns else 0
    teachers_count = len(df[df['ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ'] == 'Ν']) if 'ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ' in df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Συνολικοί Μαθητές", total_students)
    with col2:
        st.metric("Αγόρια", boys_count)
    with col3:
        st.metric("Κορίτσια", girls_count)
    with col4:
        st.metric("Παιδιά Εκπαιδευτικών", teachers_count)

def display_scenario_stats(df, scenario_col, scenario_name):
    """Εμφάνιση στατιστικών σεναρίου"""
    try:
        if scenario_col not in df.columns:
            return
            
        df_assigned = df[df[scenario_col].notna()].copy()
        if len(df_assigned) == 0:
            return
            
        df_assigned['ΤΜΗΜΑ'] = df_assigned[scenario_col]
        
        if STATS_AVAILABLE:
            stats_df = generate_statistics_table(df_assigned)
            st.subheader(f"📊 Στατιστικά {scenario_name}")
            st.dataframe(stats_df, use_container_width=True)
        else:
            # Χειροκίνητη δημιουργία στατιστικών
            st.subheader(f"📊 Στατιστικά {scenario_name}")
            stats_data = []
            for tmima in sorted(df_assigned['ΤΜΗΜΑ'].unique()):
                subset = df_assigned[df_assigned['ΤΜΗΜΑ'] == tmima]
                boys = len(subset[subset['ΦΥΛΟ'] == 'Α']) if 'ΦΥΛΟ' in subset.columns else 0
                girls = len(subset[subset['ΦΥΛΟ'] == 'Κ']) if 'ΦΥΛΟ' in subset.columns else 0
                greek = len(subset[subset['ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ'] == 'Ν']) if 'ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ' in subset.columns else 0
                total = len(subset)
                
                stats_data.append({
                    'ΤΜΗΜΑ': tmima,
                    'ΑΓΟΡΙΑ': boys,
                    'ΚΟΡΙΤΣΙΑ': girls,
                    'ΓΝΩΣΗ ΕΛΛ.': greek,
                    'ΣΥΝΟΛΟ': total
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Σφάλμα στα στατιστικά: {e}")

def run_simple_assignment(df):
    """Απλή ανάθεση χωρίς περίπλοκους αλγορίθμους"""
    try:
        st.subheader("🚀 Εκτέλεση Απλής Ανάθεσης")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Βήμα 1: Παιδιά εκπαιδευτικών
        status_text.text("Βήμα 1: Ανάθεση παιδιών εκπαιδευτικών...")
        progress_bar.progress(20)
        
        df_result = df.copy()
        df_result['ΤΜΗΜΑ'] = None
        
        # Απλή κατανομή παιδιών εκπαιδευτικών
        teacher_kids = df_result[df_result['ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ'] == 'Ν'].index.tolist()
        for i, idx in enumerate(teacher_kids):
            tmima = 'Α1' if i % 2 == 0 else 'Α2'
            df_result.loc[idx, 'ΤΜΗΜΑ'] = tmima
        
        progress_bar.progress(40)
        
        # Βήμα 2: Υπόλοιποι μαθητές
        status_text.text("Βήμα 2: Ανάθεση υπόλοιπων μαθητών...")
        
        remaining = df_result[df_result['ΤΜΗΜΑ'].isna()].index.tolist()
        
        # Ισοκατανομή με βάση το φύλο
        boys = [idx for idx in remaining if df_result.loc[idx, 'ΦΥΛΟ'] == 'Α']
        girls = [idx for idx in remaining if df_result.loc[idx, 'ΦΥΛΟ'] == 'Κ']
        
        # Κατανομή αγοριών
        for i, idx in enumerate(boys):
            current_a1 = len(df_result[(df_result['ΤΜΗΜΑ'] == 'Α1') & (df_result['ΦΥΛΟ'] == 'Α')])
            current_a2 = len(df_result[(df_result['ΤΜΗΜΑ'] == 'Α2') & (df_result['ΦΥΛΟ'] == 'Α')])
            
            if current_a1 <= current_a2:
                df_result.loc[idx, 'ΤΜΗΜΑ'] = 'Α1'
            else:
                df_result.loc[idx, 'ΤΜΗΜΑ'] = 'Α2'
        
        progress_bar.progress(60)
        
        # Κατανομή κοριτσιών
        for i, idx in enumerate(girls):
            current_a1 = len(df_result[(df_result['ΤΜΗΜΑ'] == 'Α1') & (df_result['ΦΥΛΟ'] == 'Κ')])
            current_a2 = len(df_result[(df_result['ΤΜΗΜΑ'] == 'Α2') & (df_result['ΦΥΛΟ'] == 'Κ')])
            
            if current_a1 <= current_a2:
                df_result.loc[idx, 'ΤΜΗΜΑ'] = 'Α1'
            else:
                df_result.loc[idx, 'ΤΜΗΜΑ'] = 'Α2'
        
        progress_bar.progress(80)
        
        # Τελικοποίηση
        status_text.text("Τελικοποίηση...")
        progress_bar.progress(100)
        
        status_text.text("✅ Ανάθεση ολοκληρώθηκε!")
        
        return df_result
        
    except Exception as e:
        st.error(f"Σφάλμα στην ανάθεση: {e}")
        st.code(traceback.format_exc())
        return None

def calculate_simple_score(df, tmima_col):
    """Απλός υπολογισμός score"""
    try:
        a1_data = df[df[tmima_col] == 'Α1']
        a2_data = df[df[tmima_col] == 'Α2']
        
        # Διαφορά πληθυσμού
        pop_diff = abs(len(a1_data) - len(a2_data))
        
        # Διαφορά φύλου
        a1_boys = len(a1_data[a1_data['ΦΥΛΟ'] == 'Α']) if 'ΦΥΛΟ' in df.columns else 0
        a1_girls = len(a1_data[a1_data['ΦΥΛΟ'] == 'Κ']) if 'ΦΥΛΟ' in df.columns else 0
        a2_boys = len(a2_data[a2_data['ΦΥΛΟ'] == 'Α']) if 'ΦΥΛΟ' in df.columns else 0
        a2_girls = len(a2_data[a2_data['ΦΥΛΟ'] == 'Κ']) if 'ΦΥΛΟ' in df.columns else 0
        
        boys_diff = abs(a1_boys - a2_boys)
        girls_diff = abs(a1_girls - a2_girls)
        gender_diff = max(boys_diff, girls_diff)
        
        # Διαφορά γνώσης ελληνικών
        greek_diff = 0
        if 'ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ' in df.columns:
            a1_greek = len(a1_data[a1_data['ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ'] == 'Ν'])
            a2_greek = len(a2_data[a2_data['ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ'] == 'Ν'])
            greek_diff = abs(a1_greek - a2_greek)
        
        # Συνολικό score (πολλαπλασιαστές βάσει σπουδαιότητας)
        total_score = pop_diff * 3 + gender_diff * 2 + greek_diff * 1
        
        return {
            'total_score': total_score,
            'pop_diff': pop_diff,
            'gender_diff': gender_diff,
            'greek_diff': greek_diff,
            'a1_total': len(a1_data),
            'a2_total': len(a2_data),
            'a1_boys': a1_boys,
            'a1_girls': a1_girls,
            'a2_boys': a2_boys,
            'a2_girls': a2_girls
        }
        
    except Exception as e:
        st.error(f"Σφάλμα στον υπολογισμό score: {e}")
        return None

def create_download_package(df, scenario_name="ΣΕΝΑΡΙΟ_1"):
    """Δημιουργία αρχείου download"""
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Κύριο αρχείο Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Αποτελέσματα', index=False)
                
                # Στατιστικά αν είναι διαθέσιμα
                if STATS_AVAILABLE and 'ΤΜΗΜΑ' in df.columns:
                    try:
                        df_assigned = df[df['ΤΜΗΜΑ'].notna()].copy()
                        stats_df = generate_statistics_table(df_assigned)
                        stats_df.to_excel(writer, sheet_name='Στατιστικά', index=True)
                    except:
                        pass
            
            zip_file.writestr(f"{scenario_name}_Αποτελέσματα.xlsx", excel_buffer.getvalue())
        
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Σφάλμα στη δημιουργία αρχείου: {e}")
        return None

def main():
    """Κύρια συνάρτηση"""
    init_session_state()
    
    st.title("🎓 Σύστημα Ανάθεσης Μαθητών σε Τμήματα")
    st.markdown("*Λειτουργική έκδοση με απλή ανάθεση*")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Μενού Πλοήγησης")
    
    # Upload αρχείου
    st.sidebar.subheader("📁 Φόρτωση Δεδομένων")
    uploaded_file = st.sidebar.file_uploader(
        "Επιλέξτε αρχείο Excel ή CSV",
        type=['xlsx', 'csv'],
        help="Το αρχείο πρέπει να περιέχει στήλες: ΟΝΟΜΑ, ΦΥΛΟ, ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ, ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"
    )
    
    if uploaded_file is not None:
        # Φόρτωση δεδομένων
        if st.session_state.data is None:
            with st.spinner("Φόρτωση δεδομένων..."):
                data, error = safe_load_data(uploaded_file)
                if error:
                    st.error(f"❌ {error}")
                    return
                st.session_state.data = data
                st.session_state.current_step = 1
        
        df = st.session_state.data
        
        if df is not None:
            # Εμφάνιση βασικών στοιχείων
            display_basic_info(df)
            
            # Έλεγχος στηλών
            required_cols = ['ΟΝΟΜΑ', 'ΦΥΛΟ', 'ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ', 'ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Λείπουν στήλες: {', '.join(missing_cols)}")
                st.info("Διαθέσιμες στήλες: " + ", ".join(df.columns))
            else:
                st.success("✅ Όλες οι απαιτούμενες στήλες βρέθηκαν!")
            
            # Επιλογή εκτέλεσης
            st.sidebar.subheader("🚀 Εκτέλεση Ανάθεσης")
            
            if st.sidebar.button("▶️ Εκτέλεση Ανάθεσης", disabled=bool(missing_cols)):
                with st.spinner("Εκτέλεση ανάθεσης..."):
                    result_df = run_simple_assignment(df)
                    if result_df is not None:
                        st.session_state.results['final'] = result_df
                        st.session_state.current_step = 2
            
            # Εμφάνιση αποτελεσμάτων
            if 'final' in st.session_state.results:
                st.markdown("---")
                st.subheader("🏆 Αποτελέσματα Ανάθεσης")
                
                result_df = st.session_state.results['final']
                
                # Υπολογισμός και εμφάνιση score
                score = calculate_simple_score(result_df, 'ΤΜΗΜΑ')
                if score:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Συνολικό Score", score['total_score'])
                    with col2:
                        st.metric("Διαφορά Πληθυσμού", score['pop_diff'])
                    with col3:
                        st.metric("Διαφορά Φύλου", score['gender_diff'])
                    with col4:
                        st.metric("Διαφορά Γνώσης", score['greek_diff'])
                    
                    # Αναλυτικός πίνακας αποτελεσμάτων
                    st.subheader("📊 Αναλυτικά Αποτελέσματα")
                    summary_data = [
                        {
                            'Τμήμα': 'Α1',
                            'Συνολικός Πληθυσμός': score['a1_total'],
                            'Αγόρια': score['a1_boys'],
                            'Κορίτσια': score['a1_girls']
                        },
                        {
                            'Τμήμα': 'Α2', 
                            'Συνολικός Πληθυσμός': score['a2_total'],
                            'Αγόρια': score['a2_boys'],
                            'Κορίτσια': score['a2_girls']
                        }
                    ]
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                # Στατιστικά ανά τμήμα
                display_scenario_stats(result_df, 'ΤΜΗΜΑ', 'Τελικό Σενάριο')
                
                # Εμφάνιση πλήρων αποτελεσμάτων
                with st.expander("📋 Πλήρη Αποτελέσματα"):
                    st.dataframe(result_df, use_container_width=True)
                
                # Download
                st.sidebar.subheader("💾 Λήψη Αποτελεσμάτων")
                if st.sidebar.button("📥 Δημιουργία Αρχείου"):
                    with st.spinner("Δημιουργία αρχείου..."):
                        zip_data = create_download_package(result_df)
                        if zip_data:
                            st.sidebar.download_button(
                                label="⬇️ Λήψη Αποτελεσμάτων",
                                data=zip_data,
                                file_name="Αποτελέσματα_Ανάθεσης.zip",
                                mime="application/zip"
                            )
            
            # Reset
            if st.sidebar.button("🔄 Επαναφορά"):
                st.session_state.clear()
                st.rerun()
    
    else:
        st.info("👆 Παρακαλώ ανεβάστε ένα αρχείο Excel ή CSV για να ξεκινήσετε")
        
        # Οδηγίες
        with st.expander("📖 Οδηγίες Χρήσης"):
            st.markdown("""
            ### Απαιτούμενες Στήλες:
            - **ΟΝΟΜΑ**: Ονοματεπώνυμο μαθητή
            - **ΦΥΛΟ**: Α (Αγόρι) ή Κ (Κορίτσι)
            - **ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ**: Ν (Ναι) ή Ο (Όχι)
            - **ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ**: Ν (Ναι) ή Ο (Όχι)
            
            ### Προαιρετικές Στήλες:
            - **ΦΙΛΟΙ**: Λίστα φίλων
            - **ΖΩΗΡΟΣ**: Ν/Ο
            - **ΙΔΙΑΙΤΕΡΟΤΗΤΑ**: Ν/Ο
            - **ΣΥΓΚΡΟΥΣΗ**: Λίστα συγκρουόμενων
            
            ### Τι κάνει η ανάθεση:
            1. Κατανέμει τα παιδιά εκπαιδευτικών ισοκατανομή
            2. Κατανέμει τους υπόλοιπους μαθητές με βάση το φύλο
            3. Υπολογίζει score βάσει ισορροπίας τμημάτων
            4. Δημιουργεί αρχεία Excel με αποτελέσματα
            """)

if __name__ == "__main__":
    main()
