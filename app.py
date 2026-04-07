import time
import streamlit as st
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Career Map", layout="wide")

# ---------- ONE-TIME LOADING ----------
if "loaded" not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("<h1 style='text-align: center;'>🚀 Career Map</h1>", unsafe_allow_html=True)
        st.write("Loading your career journey...")
        time.sleep(1)
    placeholder.empty()
    st.session_state.loaded = True

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_json("jobs_cleaned_final.json")
    df['combined'] = df['skill_text']
    return df

df = load_data()

# ---------- EXPERIENCE CONVERT ----------
def convert_exp(exp):
    exp = str(exp).lower()
    if "0-1" in exp:
        return 0
    elif "2-4" in exp:
        return 2
    elif "5-7" in exp:
        return 5
    else:
        return 8

df['exp_num'] = df['YearsOfExperience'].apply(lambda x: convert_exp(x))

# ---------- TF-IDF ----------
@st.cache_resource
def load_tfidf():
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df['combined'])
    return vectorizer

vectorizer = load_tfidf()

# ---------- SBERT ----------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
job_embeddings = model.encode(df['combined'].tolist())
# ---------- VALID SKILL FILTER ----------
VALID_SKILLS = {
    "python","sql","machine learning","deep learning","data science",
    "pandas","numpy","tensorflow","nlp","statistics","tableau","power bi",
    "ai","artificial intelligence","neural networks","scikit-learn","r"
}

def extract_valid_skills(text):
    text = text.lower()
    found = []
    for skill in VALID_SKILLS:
        if skill in text:
            found.append(skill)
    return list(set(found))

# ---------- HYBRID MATCH ----------
def get_matches_hybrid(user_skills, user_exp):
    user_skills = user_skills.lower()

    filtered_df = df[df['exp_num'] <= user_exp].reset_index(drop=True)

    tfidf_jobs = vectorizer.transform(filtered_df['combined'])
    tfidf_user = vectorizer.transform([user_skills])
    tfidf_scores = cosine_similarity(tfidf_user, tfidf_jobs)[0]

    sbert_jobs = job_embeddings[filtered_df.index]
    sbert_user = model.encode([user_skills])
    sbert_scores = cosine_similarity(sbert_user, sbert_jobs)[0]

    final_scores = 0.6 * sbert_scores + 0.4 * tfidf_scores
# ---------- BOOST BASED ON SKILL MATCH ----------
    for i, row in filtered_df.iterrows():
        job_text = " ".join(row['skills_list']).lower()
        user_list = [s.strip().lower() for s in user_skills.split(",")]
        job_list = [s.strip().lower() for s in row['skills_list']]
        overlap = 0
        for u in user_list:
            for j in job_list:
                if u in j or j in u:
                    overlap += 1
        final_scores[i] += overlap * 0.1
    filtered_df['score'] = final_scores * 100
    return filtered_df.sort_values(by='score', ascending=False).head(5)
# ---------- STRICT MATCH (FIXED) ----------
def get_matches_strict(user_skills_list, user_exp):

    filtered_df = df[df['exp_num'] <= user_exp].copy()

    scores = []

    for _, row in filtered_df.iterrows():
        job_skills = [s.lower() for s in row['skills_list']]

        overlap = 0

        for u in user_skills_list:
            for j in job_skills:
                if u in j or j in u:   # 🔥 KEY FIX
                    overlap += 1
        # normalize score (avoid same % issue)
        score = overlap / max(len(job_skills), 1)
        scores.append(score)



    filtered_df['strict_score'] = scores

    # remove irrelevant jobs
    filtered_df = filtered_df[filtered_df['strict_score'] > 0]

    # 🔥 REMOVE DUPLICATES
    filtered_df = filtered_df.sort_values(by='strict_score', ascending=False)
    filtered_df = filtered_df.drop_duplicates(subset=['Title'])

    return filtered_df.head(5)

# ---------- SKILL HELPERS ----------

def skill_gap(user_skills, job_skills):
    user_list = [s.strip().lower() for s in user_skills.split(",")]
    job_list = [s.lower() for s in job_skills]

    missing = []

    for j in job_list:
        found = False

        for u in user_list:
            # 🔥 STRICTER MATCH (IMPORTANT)
            if u == j or u in j:
                found = True
                break

        if not found:
            missing.append(j)

    return missing

def matched_skills(user_skills, job_skills):
    user_list = [s.strip().lower() for s in user_skills.split(",")]
    job_list = [s.lower() for s in job_skills]

    matched = []

    for j in job_list:
        for u in user_list:
            if u == j or u in j:
                matched.append(j)

    return list(set(matched))



def explain_job(user_skills, job_skills):
    matched = matched_skills(user_skills, job_skills)
    missing = skill_gap(user_skills, job_skills)

    text = f"Matches: {', '.join(matched[:3])}"
    if missing:
        text += f" | Improve: {', '.join(missing[:3])}"
    return text

# ---------- COURSES ----------
courses = {

    # ================= AI / ML =================
    "machine learning": [
        ("Coursera ML - Andrew Ng", "https://www.coursera.org/learn/machine-learning")
    ],
    "tensorflow": [
        ("TensorFlow Course", "https://www.coursera.org/professional-certificates/tensorflow-in-practice")
    ],
    "python": [
        ("Python Course", "https://www.coursera.org/specializations/python")
    ],

    # ================= .NET STACK =================
    "c#": [
        ("C# Full Course", "https://www.udemy.com/topic/c-sharp/")
    ],
    "vb.net basics": [
        ("VB.NET Course", "https://www.youtube.com/results?search_query=vb.net+course")
    ],
    ".net framework": [
        (".NET Framework Course", "https://www.youtube.com/results?search_query=.net+framework+course")
    ],
    ".net core fundamentals": [
        (".NET Core Course", "https://www.youtube.com/results?search_query=.net+core+course")
    ],
    "asp.net": [
        ("ASP.NET Course", "https://www.youtube.com/results?search_query=asp.net+course")
    ],
    "mvc": [
        ("MVC Architecture Course", "https://www.youtube.com/results?search_query=mvc+architecture")
    ],
    "entity framework basics": [
        ("Entity Framework Course", "https://www.youtube.com/results?search_query=entity+framework+course")
    ],
    "linq": [
        ("LINQ Tutorial", "https://www.youtube.com/results?search_query=linq+tutorial")
    ],

    # ================= WEB =================
    "html": [
        ("HTML Course", "https://www.youtube.com/results?search_query=html+course")
    ],
    "css": [
        ("CSS Course", "https://www.youtube.com/results?search_query=css+course")
    ],
    "javascript basics": [
        ("JavaScript Course", "https://www.youtube.com/results?search_query=javascript+course")
    ],

    # ================= DATABASE =================
    "sql server": [
        ("SQL Server Course", "https://www.youtube.com/results?search_query=sql+server+course")
    ],
    "sql basics": [
        ("SQL Basics Course", "https://www.youtube.com/results?search_query=sql+basics")
    ],
    "nosql basics": [
        ("NoSQL Course", "https://www.youtube.com/results?search_query=nosql+course")
    ],

    # ================= TOOLS =================
    "visual studio": [
        ("Visual Studio Tutorial", "https://www.youtube.com/results?search_query=visual+studio+tutorial")
    ],
    "git": [
        ("Git & GitHub Course", "https://www.youtube.com/results?search_query=git+and+github+course")
    ],
    "unit testing basics": [
        ("Unit Testing Course", "https://www.youtube.com/results?search_query=unit+testing+course")
    ],

    # ================= BIG DATA =================
    "hadoop basics": [
        ("Hadoop Course", "https://www.youtube.com/results?search_query=hadoop+course")
    ],
    "spark basics": [
        ("Apache Spark Course", "https://www.youtube.com/results?search_query=apache+spark+course")
    ],
    "scala basics": [
        ("Scala Course", "https://www.youtube.com/results?search_query=scala+course")
    ],
    "etl basics": [
        ("ETL Course", "https://www.youtube.com/results?search_query=etl+process")
    ],
    "talend basics": [
        ("Talend Tutorial", "https://www.youtube.com/results?search_query=talend+tutorial")
    ],
    "linux/unix basics": [
        ("Linux Course", "https://www.youtube.com/results?search_query=linux+basics")
    ],

    # ================= PROJECT MANAGEMENT =================
    "project planning": [
        ("Project Planning Course", "https://www.youtube.com/results?search_query=project+planning")
    ],
    "risk management": [
        ("Risk Management Course", "https://www.youtube.com/results?search_query=risk+management")
    ],
    "stakeholder management": [
        ("Stakeholder Management", "https://www.youtube.com/results?search_query=stakeholder+management")
    ],
    "team leadership": [
        ("Leadership Skills Course", "https://www.youtube.com/results?search_query=leadership+skills")
    ],
    "change management": [
        ("Change Management Course", "https://www.youtube.com/results?search_query=change+management")
    ],
    "jira": [
        ("Jira Course", "https://www.youtube.com/results?search_query=jira+tutorial")
    ],
    "ms project": [
        ("MS Project Tutorial", "https://www.youtube.com/results?search_query=ms+project+tutorial")
    ],

    # ================= SOFT SKILLS =================
    "teamwork": [
        ("Teamwork Skills", "https://www.youtube.com/results?search_query=teamwork+skills")
    ],
    "analytical thinking": [
        ("Analytical Thinking Course", "https://www.youtube.com/results?search_query=analytical+thinking")
    ],
    "curiosity": [
        ("Critical Thinking Course", "https://www.youtube.com/results?search_query=critical+thinking")
    ]
}

def recommend_courses(missing_skills):
    rec = []
    for skill in missing_skills:
        skill_lower = skill.lower()

        for key in courses:
            if key in skill_lower or skill_lower in key:
                rec += courses[key]

        # fallback → YouTube search (VERY IMPORTANT)
        if not any(skill_lower in key for key in courses):
            link = f"https://www.youtube.com/results?search_query={skill}+course"
            rec.append((f"{skill} Course", link))

    return list(set(rec))

def get_course_for_skill(skill):
    skill_lower = skill.lower()

    for key in courses:
        if key in skill_lower or skill_lower in key:
            return courses[key][0]   # return first course

    # fallback → YouTube search
    return (f"{skill} Course", f"https://www.youtube.com/results?search_query={skill}+course")




# ---------- RESUME ----------
def extract_text(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text.lower()

# ---------- UI ----------
st.markdown("<h1 style='text-align: center;'>🧭 Career Map</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered job recommendations</p>", unsafe_allow_html=True)

# ---------- FORM (NO RERUN ISSUE) ----------
with st.form("career_form"):

    # PERSONAL DETAILS
    st.subheader("👤 Personal Details")
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Full Name")
        location = st.text_input("Location")

    with col2:
        email = st.text_input("Email")
        phone = st.text_input("Phone")

    # EXPERIENCE
    st.subheader("💼 Experience")
    experience = st.selectbox("Experience", ["0-1 years", "2-4 years", "5-7 years", "8+ years"])

    # SKILLS
    st.subheader("🧠 Skills")

    skills_list = [
        "Artificial Intelligence", "Machine Learning", "Numpy", "Data Science",
        "Pandas", "Tensorflow", "Power BI", "Natural Language Processing",
        "Neural Networks", "Tableau", "Statistics", "Scikit-Learn",
        "Deep Learning", "SQL", "R", "Python"
    ]

    selected_skills = []
    cols = st.columns(4)

    for i, skill in enumerate(skills_list):
        if cols[i % 4].checkbox(skill):
            selected_skills.append(skill)

    skills_input = st.text_input("Enter skills (comma separated)")

    # COMBINE SKILLS
    final_skills_list = []

    if selected_skills:
        final_skills_list.extend(selected_skills)

    if skills_input:
        manual = [s.strip() for s in skills_input.split(",")]
        final_skills_list.extend(manual)

    final_skills_list = list(set(final_skills_list))
    final_skills = ", ".join(final_skills_list)

    # RESUME
    st.subheader("📄 Upload Resume")
    file = st.file_uploader("Upload resume", type=["pdf"])

    # SUBMIT BUTTON
    submit = st.form_submit_button("🚀 Find Jobs")

# ---------- AFTER SUBMIT ----------
if submit:
    extracted = []
    if final_skills == "" and file:
        raw_text = extract_text(file)
        extracted = extract_valid_skills(raw_text)

    if extracted:
        final_skills = ", ".join(extracted)

    if final_skills == "":
        st.warning("Please enter or select skills or upload resume")

    else:
        exp_num = convert_exp(experience)

        with st.spinner("🔍 Matching jobs using AI..."):
            # ---------- SMART MATCH SELECTION ----------
            user_skill_list = [s.strip().lower() for s in final_skills.split(",")]
            # if user selected skills → STRICT mode
            if selected_skills or skills_input:
              results = get_matches_strict(user_skill_list, exp_num)
            else:
              results = get_matches_hybrid(final_skills, exp_num)
        st.success("🎯 Top Matches Found!")
        for _, row in results.iterrows():

            gap = skill_gap(final_skills, row['skills_list'])
            matched = matched_skills(final_skills, row['skills_list'])
            explanation = explain_job(final_skills, row['skills_list'])
            course_list = recommend_courses(gap)

            st.subheader(f"🔥 {row['Title']}")
            # ---------- FIX SCORE DISPLAY ----------
            if 'score' in row:
                st.write(f"📊 Match Score: {row['score']:.2f}%")
            elif 'strict_score' in row:
                st.write(f"📊 Match Score: {row['strict_score'] * 100:.2f}%")
            st.write("✅ Matched Skills:", matched[:5])

            # ---------- SAFE MISSING SKILLS DISPLAY ----------
            if gap and len(gap) > 0:
                st.write("❌ Missing Skills & Courses:")

                for skill in gap[:8]:
                    cname, link = get_course_for_skill(skill)
                    st.markdown(f"🔸 **{skill}** → [{cname}]({link})")
            else:
                st.success("✅ No major missing skills (Great match!)")

            st.info(explanation)

            if course_list:
                st.write("📚 Recommended Courses:")
                for cname, link in course_list:
                    st.markdown(f"- [{cname}]({link})")

            st.write("---")
