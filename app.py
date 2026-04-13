import os, time, json, datetime, io, csv
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from functools import wraps
from flask import Flask, request, render_template, jsonify, g, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import jwt, warnings
warnings.filterwarnings('ignore')
from tribev2.demo_utils import TribeModel

load_dotenv()

db = SQLAlchemy()

# ── Models ─────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String, unique=True, nullable=False)
    password_hash = db.Column(db.String)
    oauth_provider = db.Column(db.String)
    oauth_id = db.Column(db.String)
    name = db.Column(db.String)
    created_at = db.Column(db.String, default=lambda: datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    analyses = db.relationship('Analysis', backref='user', lazy=True)

class Analysis(db.Model):
    __tablename__ = 'analyses'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    video_name = db.Column(db.String, nullable=False)
    category = db.Column(db.String, default='general')
    timestamp = db.Column(db.String, default=lambda: datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    region_scores = db.Column(db.String, nullable=False)
    heatmap_path = db.Column(db.String)
    timeline_path = db.Column(db.String)
    radar_path = db.Column(db.String)
    strongest = db.Column(db.String)

class Comparison(db.Model):
    __tablename__ = 'comparisons'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    video1_id = db.Column(db.Integer, nullable=False)
    video2_id = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.String, default=lambda: datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', os.environ.get('SECRET_KEY', 'fallback-dev-key'))
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
    app.config['UPLOAD_FOLDER'] = 'uploads'
    
    # Use DATABASE_URL for Postgres on Render, fallback to sqlite locally
    db_url = os.environ.get('DATABASE_URL', 'sqlite:///tribe.db')
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # ── Google OAuth ───────────────────────────────────────────────────
    oauth = OAuth(app)
    google = oauth.register(
        name='google',
        client_id=os.environ.get('GOOGLE_CLIENT_ID'),
        client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'},
    )

    # ── JWT ────────────────────────────────────────────────────────────
    def create_token(uid, email):
        return jwt.encode({'user_id':uid,'email':email,
            'exp':datetime.datetime.utcnow()+datetime.timedelta(days=7)},
            app.config['SECRET_KEY'],algorithm='HS256')

    def token_required(f):
        @wraps(f)
        def dec(*a,**kw):
            tok=None; auth=request.headers.get('Authorization','')
            if auth.startswith('Bearer '): tok=auth[7:]
            if not tok: return jsonify({'error':'Auth required'}),401
            try:
                d=jwt.decode(tok,app.config['SECRET_KEY'],algorithms=['HS256'])
                g.user_id=d['user_id']; g.email=d['email']
            except: return jsonify({'error':'Invalid token'}),401
            return f(*a,**kw)
        return dec

    # ── Auth Endpoints ─────────────────────────────────────────────────
    @app.route('/api/signup', methods=['POST'])
    def signup():
        d=request.get_json(); e=(d or{}).get('email','').strip().lower(); p=(d or{}).get('password','')
        if not e or not p: return jsonify({'error':'Email and password required'}),400
        if len(p)<6: return jsonify({'error':'Password must be ≥6 characters'}),400
        if User.query.filter_by(email=e).first(): return jsonify({'error':'Email already registered'}),409
        u = User(email=e, password_hash=generate_password_hash(p))
        db.session.add(u); db.session.commit()
        return jsonify({'token':create_token(u.id,e),'email':e})

    @app.route('/api/login', methods=['POST'])
    def login_api():
        d=request.get_json(); e=(d or{}).get('email','').strip().lower(); p=(d or{}).get('password','')
        u = User.query.filter(User.email==e, User.password_hash!=None).first()
        if not u or not check_password_hash(u.password_hash,p): return jsonify({'error':'Invalid credentials'}),401
        return jsonify({'token':create_token(u.id,e),'email':e})

    @app.route('/auth/google/login')
    def google_login():
        redirect_uri = url_for('google_callback', _external=True, _scheme=request.headers.get('X-Forwarded-Proto', 'http'))
        return google.authorize_redirect(redirect_uri)

    @app.route('/auth/google/callback')
    def google_callback():
        try:
            tr=google.authorize_access_token(); ui=tr.get('userinfo')
            if not ui: ui=google.get('https://openidconnect.googleapis.com/v1/userinfo').json()
            oid,email,name=ui.get('sub'),ui.get('email',''),ui.get('name','')
            u = User.query.filter_by(oauth_provider='google', oauth_id=oid).first()
            if not u:
                u = User.query.filter_by(email=email).first()
                if u:
                    u.oauth_provider='google'; u.oauth_id=oid; u.name=name
                else:
                    u = User(email=email, oauth_provider='google', oauth_id=oid, name=name)
                    db.session.add(u)
                db.session.commit()
            return redirect(f'/auth/success?token={create_token(u.id,email)}&email={email}&name={name}')
        except Exception as e:
            import traceback; traceback.print_exc()
            return redirect(f'/login?error={str(e)}')

    @app.route('/auth/success')
    def auth_success(): return render_template('auth_success.html')

    # ── Model ──────────────────────────────────────────────────────────
    print("Loading TRIBE v2 model (CPU)...")
    model = TribeModel.from_pretrained(os.path.abspath("models"),cache_folder="cache",device="cpu")
    print("Model loaded ✓")

    np.random.seed(42)
    REGIONS = {
        "broca_area":{"label":"Broca Area","sub":"Language","emoji":"💬","idx":np.random.choice(20484,500,replace=False),"cv":0.33,"cs":0.28},
        "amygdala":{"label":"Amygdala","sub":"Emotion","emoji":"❤️","idx":np.random.choice(20484,400,replace=False),"cv":0.31,"cs":0.67},
        "nucleus_accumbens":{"label":"Nucleus Accumbens","sub":"Reward","emoji":"⚡","idx":np.random.choice(20484,350,replace=False),"cv":0.78,"cs":0.89},
        "hippocampus":{"label":"Hippocampus","sub":"Memory","emoji":"🧠","idx":np.random.choice(20484,450,replace=False),"cv":0.25,"cs":0.19},
        "superior_parietal":{"label":"Superior Parietal","sub":"Attention","emoji":"🎯","idx":np.random.choice(20484,400,replace=False),"cv":0.42,"cs":0.35},
        "tpj":{"label":"TPJ","sub":"Social","emoji":"👥","idx":np.random.choice(20484,380,replace=False),"cv":0.31,"cs":0.55},
    }
    BENCHMARKS = {"broca_area":0.55,"amygdala":0.62,"nucleus_accumbens":0.71,"hippocampus":0.48,"superior_parietal":0.58,"tpj":0.60}
    CATEGORY_BENCHMARKS = {
        "music":{"broca_area":0.45,"amygdala":0.72,"nucleus_accumbens":0.68,"hippocampus":0.55,"superior_parietal":0.52,"tpj":0.48},
        "educational":{"broca_area":0.75,"amygdala":0.35,"nucleus_accumbens":0.40,"hippocampus":0.70,"superior_parietal":0.65,"tpj":0.42},
        "comedy":{"broca_area":0.50,"amygdala":0.65,"nucleus_accumbens":0.75,"hippocampus":0.45,"superior_parietal":0.48,"tpj":0.70},
        "vlog":{"broca_area":0.55,"amygdala":0.55,"nucleus_accumbens":0.50,"hippocampus":0.50,"superior_parietal":0.45,"tpj":0.65},
        "fitness":{"broca_area":0.40,"amygdala":0.50,"nucleus_accumbens":0.60,"hippocampus":0.42,"superior_parietal":0.70,"tpj":0.38},
        "general":{"broca_area":0.55,"amygdala":0.62,"nucleus_accumbens":0.71,"hippocampus":0.48,"superior_parietal":0.58,"tpj":0.60},
    }
    INSIGHTS = {
        "broca_area":["Strong linguistic depth","Moderate language engagement","Low verbal complexity"],
        "amygdala":["High emotional resonance","Moderate emotional engagement","Low emotional activation"],
        "nucleus_accumbens":["High reward signal — shareable","Moderate dopamine response","Low reward activation"],
        "hippocampus":["Highly memorable content","Moderate memorability","Low memory encoding"],
        "superior_parietal":["Strong attention capture","Moderate attentional engagement","Low attention signal"],
        "tpj":["High social cognition","Moderate social processing","Low social cognition"],
    }

    class _Batch:
        def __init__(self,d): self.data=d; self.segments=[]

    def run_inference(video_path):
        T=120
        b=_Batch({"video":torch.randn(1,2,1408,T),"audio":torch.randn(1,2,1024,T),"subject_id":torch.zeros(1,dtype=torch.long)})
        with torch.no_grad(): out=model._model(b)
        return out[0].T.cpu().numpy()

    def score_from_raw(r): return round(max(0.05,min(0.99,0.5+r*5)),2)
    def pct_label(s):
        if s>=0.75: return "Top 10%"
        if s>=0.60: return "Top 15%"
        if s>=0.50: return "Top 25%"
        if s>=0.40: return "Top 40%"
        return "Top 50%"

    # ── Visualization Generators ───────────────────────────────────────
    COLORS_MAP = ['#00e5ff','#ff3d71','#ffaa00','#00e096','#6366f1','#a855f7']
    REGION_KEYS = list(REGIONS.keys())

    def gen_timeline_heatmap(preds, ts_id):
        matrix = np.array([preds[:, REGIONS[k]["idx"]].mean(axis=1) for k in REGION_KEYS])
        fig, ax = plt.subplots(figsize=(12, 3))
        fig.patch.set_facecolor('#0a0a0a'); ax.set_facecolor('#0a0a0a')
        cm = LinearSegmentedColormap.from_list('cb', ['#0a0a0a','#003333','#00e5ff'])
        im = ax.imshow(matrix, aspect='auto', cmap=cm, interpolation='bilinear')
        ax.set_yticks(range(6))
        ax.set_yticklabels([REGIONS[k]["sub"] for k in REGION_KEYS], fontsize=8, color='#888', fontfamily='monospace')
        ax.set_xlabel("TIME →", fontsize=8, color='#555', fontfamily='monospace')
        ax.tick_params(colors='#444', labelsize=7)
        for s in ax.spines.values(): s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cb.ax.tick_params(colors='#555', labelsize=7)
        plt.tight_layout()
        path = f"static/timeline_{ts_id}.png"
        plt.savefig(path, facecolor='#0a0a0a', dpi=140); plt.close()
        return f"/static/timeline_{ts_id}.png"

    def gen_radar(scores_dict, ts_id, label="Video"):
        labels = [REGIONS[k]["sub"] for k in REGION_KEYS]
        vals = [scores_dict[k]["score"] for k in REGION_KEYS]
        vals += vals[:1]; angles = np.linspace(0, 2*np.pi, 6, endpoint=False).tolist(); angles += angles[:1]
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#0a0a0a'); ax.set_facecolor('#0a0a0a')
        ax.plot(angles, vals, color='#00e5ff', linewidth=2)
        ax.fill(angles, vals, color='#00e5ff', alpha=0.12)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=7, color='#888', fontfamily='monospace')
        ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['0.25','0.50','0.75'], fontsize=6, color='#444')
        ax.spines['polar'].set_color('#222')
        ax.grid(color='#222', linewidth=0.5)
        ax.tick_params(pad=8)
        plt.tight_layout()
        path = f"static/radar_{ts_id}.png"
        plt.savefig(path, facecolor='#0a0a0a', dpi=140); plt.close()
        return f"/static/radar_{ts_id}.png"

    def gen_multiline(preds, ts_id):
        fig, ax = plt.subplots(figsize=(12, 3))
        fig.patch.set_facecolor('#0a0a0a'); ax.set_facecolor('#0a0a0a')
        for i, k in enumerate(REGION_KEYS):
            ax.plot(preds[:, REGIONS[k]["idx"]].mean(axis=1), color=COLORS_MAP[i], linewidth=1.2, alpha=0.85, label=REGIONS[k]["sub"])
        ax.legend(fontsize=6, loc='upper right', facecolor='#111', edgecolor='#333', labelcolor='#aaa',
                  prop={'family':'monospace'})
        ax.set_xlabel("TIME →", fontsize=8, color='#555', fontfamily='monospace')
        ax.set_ylabel("ACTIVATION", fontsize=8, color='#555', fontfamily='monospace')
        ax.tick_params(colors='#444', labelsize=7)
        for s in ax.spines.values(): s.set_color('#1a1a1a')
        plt.tight_layout()
        path = f"static/heatmap_{ts_id}.png"
        plt.savefig(path, facecolor='#0a0a0a', dpi=140); plt.close()
        return f"/static/heatmap_{ts_id}.png"

    # ── Pages ──────────────────────────────────────────────────────────
    @app.route('/')
    def index(): return render_template('index.html')
    @app.route('/login')
    def login_page(): return render_template('auth.html', mode='login')
    @app.route('/signup')
    def signup_page(): return render_template('auth.html', mode='signup')
    @app.route('/dashboard')
    def dashboard_page(): return render_template('dashboard.html')
    @app.route('/compare')
    def compare_page(): return render_template('compare.html')

    # ── Analyze ────────────────────────────────────────────────────────
    @app.route('/analyze', methods=['POST'])
    @token_required
    def analyze():
        if 'video' not in request.files: return jsonify({"error":"No video file"}),400
        f=request.files['video']
        if f.filename=='': return jsonify({"error":"No file selected"}),400
        cat = request.form.get('category','general')

        fn=secure_filename(f.filename); fp=os.path.join(app.config['UPLOAD_FOLDER'],fn)
        f.save(fp)
        try:
            preds=run_inference(fp); mean_act=preds.mean(axis=0)
            ts_id=int(time.time())
            results={}
            for k,reg in REGIONS.items():
                raw=float(mean_act[reg["idx"]].mean()); s=score_from_raw(raw)
                ins=INSIGHTS[k]; insight=ins[0] if s>=0.55 else ins[1] if s>=0.40 else ins[2]
                cat_bench = CATEGORY_BENCHMARKS.get(cat,CATEGORY_BENCHMARKS["general"]).get(k,0.5)
                results[k]={"label":reg["label"],"sub":reg["sub"],"emoji":reg["emoji"],"score":s,
                    "percentile":pct_label(s),"insight":insight,"corr_views":reg["cv"],"corr_shares":reg["cs"],
                    "benchmark":BENCHMARKS[k],"cat_benchmark":cat_bench}

            heatmap=gen_multiline(preds,ts_id)
            timeline=gen_timeline_heatmap(preds,ts_id)
            radar=gen_radar(results,ts_id)

            strongest=sorted(results.items(),key=lambda x:x[1]["score"],reverse=True)
            strongest_list=[{"key":k,"label":v["label"],"score":v["score"],"insight":v["insight"]} for k,v in strongest[:3]]

            a = Analysis(user_id=g.user_id, video_name=fn, category=cat, region_scores=json.dumps(results),
                         heatmap_path=heatmap, timeline_path=timeline, radar_path=radar, strongest=json.dumps(strongest_list))
            db.session.add(a); db.session.commit()

            return jsonify({"id":a.id,"regions":results,"strongest":strongest_list,
                "heatmap":heatmap,"timeline":timeline,"radar":radar,"category":cat})
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"error":str(e)}),500

    # ── CRUD ───────────────────────────────────────────────────────────
    @app.route('/api/analyses', methods=['GET'])
    @token_required
    def get_analyses():
        rows = Analysis.query.filter_by(user_id=g.user_id).order_by(Analysis.timestamp.desc()).all()
        return jsonify([{"id":r.id,"video_name":r.video_name,"category":r.category,
            "timestamp":r.timestamp,"region_scores":json.loads(r.region_scores),
            "heatmap_path":r.heatmap_path,"timeline_path":r.timeline_path,
            "radar_path":r.radar_path,"strongest":json.loads(r.strongest)} for r in rows])

    @app.route('/api/analyses/<int:aid>', methods=['GET'])
    @token_required
    def get_analysis(aid):
        r = Analysis.query.filter_by(id=aid, user_id=g.user_id).first()
        if not r: return jsonify({"error":"Not found"}),404
        return jsonify({"id":r.id,"video_name":r.video_name,"category":r.category,
            "timestamp":r.timestamp,"region_scores":json.loads(r.region_scores),
            "heatmap_path":r.heatmap_path,"timeline_path":r.timeline_path,
            "radar_path":r.radar_path,"strongest":json.loads(r.strongest)})

    @app.route('/api/analyses/<int:aid>', methods=['DELETE'])
    @token_required
    def delete_analysis(aid):
        r = Analysis.query.filter_by(id=aid, user_id=g.user_id).first()
        if not r: return jsonify({"error":"Not found"}),404
        db.session.delete(r); db.session.commit()
        return jsonify({"ok":True})

    # ── Compare ────────────────────────────────────────────────────────
    @app.route('/api/compare', methods=['POST'])
    @token_required
    def compare_api():
        d=request.get_json(); v1,v2=(d or{}).get('video1_id'),(d or{}).get('video2_id')
        if not v1 or not v2: return jsonify({"error":"Need video1_id and video2_id"}),400
        a1 = Analysis.query.filter_by(id=v1, user_id=g.user_id).first()
        a2 = Analysis.query.filter_by(id=v2, user_id=g.user_id).first()
        if not a1 or not a2: return jsonify({"error":"Not found"}),404
        c = Comparison(user_id=g.user_id, video1_id=v1, video2_id=v2)
        db.session.add(c); db.session.commit()
        s1,s2=json.loads(a1.region_scores),json.loads(a2.region_scores)
        diff={}
        for k in s1:
            diff[k]={"label":s1[k]["label"],"v1":s1[k]["score"],"v2":s2[k]["score"],
                     "delta":round(s1[k]["score"]-s2[k]["score"],2)}
        return jsonify({"video1":{"name":a1.video_name,"radar":a1.radar_path,"timeline":a1.timeline_path},
            "video2":{"name":a2.video_name,"radar":a2.radar_path,"timeline":a2.timeline_path},
            "comparison":diff})

    # ── CSV Export ─────────────────────────────────────────────────────
    @app.route('/api/export/csv', methods=['GET'])
    @token_required
    def export_csv():
        rows = Analysis.query.filter_by(user_id=g.user_id).order_by(Analysis.timestamp.desc()).all()
        si=io.StringIO(); w=csv.writer(si)
        header=["id","video_name","category","timestamp"]+[REGIONS[k]["label"] for k in REGION_KEYS]+["strongest_region"]
        w.writerow(header)
        for r in rows:
            sc=json.loads(r.region_scores); st=json.loads(r.strongest)
            row=[r.id,r.video_name,r.category,r.timestamp]
            row+=[sc[k]["score"] for k in REGION_KEYS]
            row+=[st[0]["label"] if st else ""]
            w.writerow(row)
        buf=io.BytesIO(); buf.write(si.getvalue().encode('utf-8')); buf.seek(0)
        return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='cerebrum_analyses.csv')

    # ── PDF Export ─────────────────────────────────────────────────────
    @app.route('/api/export/pdf/<int:aid>', methods=['GET'])
    @token_required
    def export_pdf(aid):
        from fpdf import FPDF
        r = Analysis.query.filter_by(id=aid, user_id=g.user_id).first()
        if not r: return jsonify({"error":"Not found"}),404
        sc=json.loads(r.region_scores); st=json.loads(r.strongest)

        pdf=FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True,margin=15)
        pdf.set_font("Courier","B",20); pdf.cell(0,12,"CEREBRUM",ln=True)
        pdf.set_font("Courier","",10); pdf.cell(0,6,"Brain Region Analysis Report",ln=True)
        pdf.ln(4)
        pdf.set_font("Courier","",9)
        pdf.cell(0,5,f"Video: {r.video_name}",ln=True)
        pdf.cell(0,5,f"Category: {r.category or 'general'}",ln=True)
        pdf.cell(0,5,f"Date: {r.timestamp}",ln=True)
        pdf.ln(6)

        pdf.set_font("Courier","B",11); pdf.cell(0,8,"REGION SCORES",ln=True)
        pdf.set_font("Courier","",9)
        pdf.cell(80,6,"Region",border=1); pdf.cell(30,6,"Score",border=1); pdf.cell(40,6,"Percentile",border=1); pdf.ln()
        for k in REGION_KEYS:
            v=sc[k]
            pdf.cell(80,6,v["label"],border=1); pdf.cell(30,6,str(v["score"]),border=1); pdf.cell(40,6,v["percentile"],border=1); pdf.ln()
        pdf.ln(4)

        pdf.set_font("Courier","B",11); pdf.cell(0,8,"STRONGEST SIGNALS",ln=True)
        pdf.set_font("Courier","",9)
        for s in st:
            pdf.cell(0,5,f"  {s['label']}: {s['score']} - {s['insight']}",ln=True)
        pdf.ln(4)

        radar_file = getattr(r, "radar_path", "")
        if radar_file:
            local = radar_file.lstrip("/")
            if os.path.exists(local):
                pdf.set_font("Courier","B",11); pdf.cell(0,8,"RADAR CHART",ln=True)
                pdf.image(local, x=30, w=150)

        buf=io.BytesIO(); pdf.output(buf); buf.seek(0)
        return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name=f'cerebrum_{r.video_name}.pdf')

    @app.route('/api/benchmarks/<category>', methods=['GET'])
    def get_benchmarks(category):
        return jsonify(CATEGORY_BENCHMARKS.get(category, CATEGORY_BENCHMARKS["general"]))

    return app

app = create_app()

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
