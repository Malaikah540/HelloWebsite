const CLASSES = [
  { id: 1, subject: 'Robotics Fundamentals',  grade: 'Grade 7A', days: 'Mon / Wed / Fri', time: '08:00', end: '09:00', room: 'Lab 3',  students: generateStudents(1, 18) },
  { id: 2, subject: 'AI & Coding',            grade: 'Grade 8B', days: 'Tue / Thu',       time: '09:30', end: '10:30', room: 'Lab 1',  students: generateStudents(2, 14) },
  { id: 3, subject: 'Electronics Workshop',   grade: 'Grade 6C', days: 'Mon / Wed',       time: '11:00', end: '12:00', room: 'Lab 2',  students: generateStudents(3, 16) },
  { id: 4, subject: 'Computational Thinking', grade: 'Grade 9A', days: 'Tue / Fri',       time: '13:00', end: '14:00', room: 'Room 5', students: generateStudents(4, 10) },
];

const NAMES = ['Ahmed Raza','Fatima Khan','Usman Tariq','Ayesha Mir','Bilal Shah','Zara Iqbal','Hassan Ali','Maryam Butt','Omar Cheema','Hina Siddiqui','Saad Malik','Noor Baig','Kamran Javed','Sara Naz','Imran Qureshi','Layla Rehman','Tariq Mehmood','Amna Arif','Asad Riaz','Dua Farooq'];
const AVATAR_COLORS = ['#10b981','#3b82f6','#8b5cf6','#f59e0b','#ef4444','#06b6d4','#ec4899'];
const EMOTIONS = ['happy','happy','happy','confused','confused','angry'];

function generateStudents(classId, count) {
  const shuffled = [...NAMES].sort(() => Math.random() - .5);
  return shuffled.slice(0, count).map((name, i) => ({
    id: 'S' + classId + String(i+1).padStart(2,'0'),
    name,
    emotion: EMOTIONS[Math.floor(Math.random() * EMOTIONS.length)],
    avatarColor: AVATAR_COLORS[i % AVATAR_COLORS.length],
    initials: name.split(' ').map(w => w[0]).join(''),
    note: '',
  }));
}

let activeFilter = 'all';
let currentClassId = null;

function updateClock() {
  const now = new Date();
  document.getElementById('liveClock').textContent = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  document.getElementById('liveDate').textContent   = now.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
}
setInterval(updateClock, 1000);
updateClock();

function isLive(cls) {
  const now = new Date();
  const [sh, sm] = cls.time.split(':').map(Number);
  const [eh, em] = cls.end.split(':').map(Number);
  const start = new Date(now); start.setHours(sh, sm, 0);
  const end   = new Date(now); end.setHours(eh, em, 0);
  return now >= start && now <= end;
}

function switchView(viewId, linkEl) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('view-' + viewId).classList.add('active');
  if (linkEl) linkEl.classList.add('active');
  if (viewId === 'overview')  renderOverview();
  if (viewId === 'classes')   renderClasses();
  if (viewId === 'students')  renderAllStudents();
  if (viewId === 'analytics') renderAnalytics();
}

function renderOverview() {
  renderSchedule();
  renderMoodBars();
  const all = CLASSES.flatMap(c => c.students);
  document.getElementById('overviewConfused').textContent = all.filter(s => s.emotion !== 'happy').length;
}

function renderSchedule() {
  document.getElementById('overviewSchedule').innerHTML = CLASSES.map(c => {
    const live = isLive(c);
    return '<li class="schedule-item' + (live ? ' active-class' : '') + '" onclick="openClass(' + c.id + ')">'
      + '<div class="schedule-time">' + c.time + ' - ' + c.end + '</div>'
      + '<div><div class="schedule-name">' + c.subject + '</div><div class="schedule-grade">' + c.grade + ' - ' + c.room + '</div></div>'
      + (live ? '<span class="live-badge"><span class="blink-dot green"></span>LIVE</span>' : '')
      + '</li>';
  }).join('');
}

function renderMoodBars() {
  const all = CLASSES.flatMap(c => c.students);
  const total = all.length;
  const rows = [
    { label: 'Happy',    count: all.filter(s => s.emotion === 'happy').length,    color: '#10b981' },
    { label: 'Confused', count: all.filter(s => s.emotion === 'confused').length, color: '#f59e0b' },
    { label: 'Angry',    count: all.filter(s => s.emotion === 'angry').length,    color: '#ef4444' },
  ];
  document.getElementById('moodBars').innerHTML = rows.map(r => {
    const pct = total ? Math.round((r.count / total) * 100) : 0;
    return '<div class="mood-row"><span class="mood-label">' + r.label + '</span>'
      + '<div class="mood-bar-bg"><div class="mood-bar-fill" style="width:' + pct + '%;background:' + r.color + '"></div></div>'
      + '<span class="mood-val" style="color:' + r.color + '">' + pct + '%</span></div>';
  }).join('');
}

function renderClasses() {
  document.getElementById('classesGrid').innerHTML = CLASSES.map(c => {
    const h = c.students.filter(s => s.emotion === 'happy').length;
    const cf = c.students.filter(s => s.emotion === 'confused').length;
    const a = c.students.filter(s => s.emotion === 'angry').length;
    const live = isLive(c);
    return '<div class="class-card' + (live ? ' live-card' : '') + '" onclick="openClass(' + c.id + ')">'
      + '<div class="class-card-header"><div><div class="class-subject">' + c.subject + '</div><div class="class-grade">' + c.grade + '</div></div>'
      + (live ? '<span class="live-badge"><span class="blink-dot green"></span>LIVE</span>' : '') + '</div>'
      + '<div class="class-time-row"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
      + c.days + ' - ' + c.time + ' to ' + c.end + ' - ' + c.room + '</div>'
      + '<div class="class-stats">'
      + '<div class="class-stat"><div class="class-stat-num g">' + h + '</div><div class="class-stat-label">Happy</div></div>'
      + '<div class="class-stat"><div class="class-stat-num y">' + cf + '</div><div class="class-stat-label">Confused</div></div>'
      + '<div class="class-stat"><div class="class-stat-num r">' + a + '</div><div class="class-stat-label">Angry</div></div>'
      + '<div class="class-stat"><div class="class-stat-num" style="color:#3b82f6">' + c.students.length + '</div><div class="class-stat-label">Students</div></div>'
      + '</div></div>';
  }).join('');
}

function openClass(classId) {
  currentClassId = classId;
  const cls = CLASSES.find(c => c.id === classId);
  document.getElementById('detailClassName').textContent = cls.subject + ' - ' + cls.grade;
  document.getElementById('detailClassMeta').textContent = cls.days + '  -  ' + cls.time + ' to ' + cls.end + '  -  ' + cls.room;
  document.getElementById('detailLive').style.display = isLive(cls) ? 'flex' : 'none';
  activeFilter = 'all';
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  document.querySelector('.chip').classList.add('active');
  document.getElementById('studentSearch').value = '';
  renderStudentList(cls.students);
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-classDetail').classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelector('[data-view=classes]').classList.add('active');
}

function renderStudentList(students) {
  const query = (document.getElementById('studentSearch') ? document.getElementById('studentSearch').value : '').toLowerCase();
  const filtered = students.filter(s =>
    (activeFilter === 'all' || s.emotion === activeFilter) &&
    (s.name.toLowerCase().includes(query) || s.id.toLowerCase().includes(query))
  );
  document.getElementById('studentList').innerHTML = filtered.length
    ? filtered.map(studentCard).join('')
    : '<p style="color:var(--muted);font-size:.88rem;padding:20px 0">No students match your filter.</p>';
}

function studentCard(s) {
  const emoji  = { happy: '😊', confused: '😕', angry: '😤' };
  const color  = { happy: 'green', confused: 'yellow', angry: 'red' };
  const action = { happy: { cls: 'view', label: 'View' }, confused: { cls: 'intervene', label: 'Help' }, angry: { cls: 'calm', label: 'Calm' } };
  const a = action[s.emotion];
  return '<div class="student-card ' + s.emotion + '" id="card-' + s.id + '">'
    + '<div class="student-avatar" style="background:' + s.avatarColor + '">' + s.initials + '</div>'
    + '<div class="student-info"><div class="student-name">' + s.name + '</div><div class="student-id">' + s.id + '</div></div>'
    + '<span class="emotion-badge ' + s.emotion + '"><span class="blink-dot ' + color[s.emotion] + '"></span>' + emoji[s.emotion] + ' ' + s.emotion.charAt(0).toUpperCase() + s.emotion.slice(1) + '</span>'
    + '<button class="student-action ' + a.cls + '" onclick="openPanel(\'' + s.id + '\',' + currentClassId + ')">' + a.label + '</button>'
    + '</div>';
}

function filterStudents() {
  const cls = CLASSES.find(c => c.id === currentClassId);
  if (cls) renderStudentList(cls.students);
}

function filterByEmotion(emotion, btn) {
  activeFilter = emotion;
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  filterStudents();
}

function renderAllStudents() {
  const all = CLASSES.flatMap(c => c.students.map(s => Object.assign({}, s, { classLabel: c.subject + ' - ' + c.grade })));
  const emap = { happy: '😊 Happy', confused: '😕 Confused', angry: '😤 Angry' };
  const cmap = { happy: 'green', confused: 'yellow', angry: 'red' };
  document.getElementById('allStudentsList').innerHTML = all.map(s =>
    '<div class="student-card ' + s.emotion + '">'
    + '<div class="student-avatar" style="background:' + s.avatarColor + '">' + s.initials + '</div>'
    + '<div class="student-info"><div class="student-name">' + s.name + '</div><div class="student-id">' + s.id + ' - <span style="color:var(--muted)">' + s.classLabel + '</span></div></div>'
    + '<span class="emotion-badge ' + s.emotion + '"><span class="blink-dot ' + cmap[s.emotion] + '"></span>' + emap[s.emotion] + '</span>'
    + '</div>'
  ).join('');
}

const INTERVENTIONS = {
  confused: [
    { icon: '🎯', bg: '#fef3c7', title: 'Change to Easier Activity', desc: 'Switch to a simplified version of the current task.' },
    { icon: '🖼', bg: '#dbeafe', title: 'Show Visual Aids',           desc: 'Display diagrams or step-by-step visuals for the topic.' },
    { icon: '🤖', bg: '#f3e8ff', title: 'GrowBot Mini-Lesson',         desc: 'Trigger a short interactive bot explanation for the student.' },
    { icon: '👥', bg: '#dcfce7', title: 'Pair with Peer',               desc: 'Seat the student next to a confident classmate.' },
    { icon: '✋', bg: '#fce7f3', title: 'One-on-One Check-in',          desc: 'Pause and speak to the student privately.' },
  ],
  angry: [
    { icon: '😌', bg: '#dcfce7', title: 'Offer a Short Break',          desc: 'Let the student step away for 2-3 minutes.' },
    { icon: '🎮', bg: '#dbeafe', title: 'Switch to Fun Activity',        desc: 'Redirect to a lighter, gamified GrowBot challenge.' },
    { icon: '✋', bg: '#fce7f3', title: 'Private Conversation',          desc: 'Gently ask the student what is bothering them.' },
    { icon: '🎵', bg: '#f3e8ff', title: 'Calm-Down Exercise',            desc: 'Play a quick breathing / calm-down GrowBot animation.' },
  ],
  happy: [
    { icon: '⭐', bg: '#fef3c7', title: 'Award GrowBot Points',         desc: 'Recognise the student\'s great attitude with bonus points.' },
    { icon: '🚀', bg: '#dcfce7', title: 'Extension Challenge',           desc: 'Give an advanced bonus task to keep them engaged.' },
    { icon: '📣', bg: '#dbeafe', title: 'Peer Mentor Role',              desc: 'Ask them to assist a classmate who is struggling.' },
  ],
};

function openPanel(studentId, classId) {
  const cls = CLASSES.find(c => c.id === classId);
  const student = cls ? cls.students.find(s => s.id === studentId) : null;
  if (!student) return;

  document.getElementById('panelStudentName').textContent = student.name;

  const styleMap = {
    happy:    { bg: '#dcfce7', color: '#065f46', label: '😊 Happy & Engaged' },
    confused: { bg: '#fef3c7', color: '#92400e', label: '😕 Confused' },
    angry:    { bg: '#fee2e2', color: '#991b1b', label: '😤 Frustrated' },
  };
  const st = styleMap[student.emotion];
  const badge = document.getElementById('panelEmotionBadge');
  badge.textContent = st.label; badge.style.background = st.bg; badge.style.color = st.color;

  const detailText = {
    happy:    'This student appears happy and engaged. Reinforce their positive attitude and consider giving them an extra challenge.',
    confused: 'This student is showing signs of confusion. Early intervention can prevent frustration from building up.',
    angry:    'This student appears frustrated or upset. A calm, empathetic approach works best here.',
  };
  document.getElementById('emotionDetail').textContent = detailText[student.emotion];

  document.getElementById('interventionActions').innerHTML = (INTERVENTIONS[student.emotion] || []).map(a =>
    '<div class="action-card" onclick="applyIntervention(\'' + a.title.replace(/'/g, "\\'") + '\')">'
    + '<div class="action-icon" style="background:' + a.bg + '">' + a.icon + '</div>'
    + '<div><div class="action-title">' + a.title + '</div><div class="action-desc">' + a.desc + '</div></div>'
    + '</div>'
  ).join('');

  const noteEl = document.getElementById('quickNote');
  noteEl.value = student.note || '';
  noteEl.dataset.studentId = studentId;
  noteEl.dataset.classId   = classId;

  document.getElementById('interventionPanel').classList.add('open');
  document.getElementById('panelOverlay').classList.add('open');
}

function closePanel() {
  document.getElementById('interventionPanel').classList.remove('open');
  document.getElementById('panelOverlay').classList.remove('open');
}

function applyIntervention(title) {
  showToast('Applied: ' + title);
  closePanel();
}

function saveNote() {
  const el = document.getElementById('quickNote');
  const cls = CLASSES.find(c => c.id === Number(el.dataset.classId));
  const student = cls ? cls.students.find(s => s.id === el.dataset.studentId) : null;
  if (student) { student.note = el.value; showToast('Note saved'); }
}

function renderAnalytics() {
  const rates = [
    { label: 'Robotics - 7A',        rate: 82 },
    { label: 'AI & Coding - 8B',     rate: 74 },
    { label: 'Electronics - 6C',     rate: 68 },
    { label: 'Comp. Thinking - 9A',  rate: 91 },
  ];
  document.getElementById('engagementBars').innerHTML = rates.map(r =>
    '<div class="eng-row"><div class="eng-label"><span>' + r.label + '</span><span style="color:var(--green)">' + r.rate + '%</span></div>'
    + '<div class="eng-bar-bg"><div class="eng-bar-fill" style="width:' + r.rate + '%"></div></div></div>'
  ).join('');

  const all = CLASSES.flatMap(c => c.students.filter(s => s.emotion !== 'happy').map(s => Object.assign({}, s, { classLabel: c.subject + ' - ' + c.grade, classId: c.id })));
  const cmap = { confused: 'yellow', angry: 'red' };
  const emap = { confused: '😕 Confused', angry: '😤 Angry' };
  const amap = { confused: 'intervene', angry: 'calm' };
  document.getElementById('followupList').innerHTML = all.map(s =>
    '<div class="student-card ' + s.emotion + '">'
    + '<div class="student-avatar" style="background:' + s.avatarColor + '">' + s.initials + '</div>'
    + '<div class="student-info"><div class="student-name">' + s.name + '</div><div class="student-id">' + s.id + ' - <span style="color:var(--muted)">' + s.classLabel + '</span></div></div>'
    + '<span class="emotion-badge ' + s.emotion + '"><span class="blink-dot ' + cmap[s.emotion] + '"></span>' + emap[s.emotion] + '</span>'
    + '<button class="student-action ' + amap[s.emotion] + '" onclick="openPanel(\'' + s.id + '\',' + s.classId + ')">Help</button>'
    + '</div>'
  ).join('');

  drawChart();
}

function drawChart() {
  const canvas = document.getElementById('emotionChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const days = ['Mon','Tue','Wed','Thu','Fri'];
  const data = { happy: [65,70,60,75,80], confused: [25,20,28,15,12], angry: [10,10,8,5,8] };
  const pL=40, pR=20, pT=20, pB=36;
  const w=W-pL-pR, h=H-pT-pB, step=w/(days.length-1);
  const gx = i => pL+i*step;
  const gy = v => pT+h-(v/100)*h;

  ctx.strokeStyle='#f1f5f9'; ctx.lineWidth=1;
  [0,25,50,75,100].forEach(v => {
    ctx.beginPath(); ctx.moveTo(pL,gy(v)); ctx.lineTo(W-pR,gy(v)); ctx.stroke();
    ctx.fillStyle='#9ca3af'; ctx.font='11px Inter,sans-serif'; ctx.textAlign='left';
    ctx.fillText(v+'%', 0, gy(v)+4);
  });
  days.forEach((d,i) => { ctx.fillStyle='#9ca3af'; ctx.font='11px Inter,sans-serif'; ctx.textAlign='center'; ctx.fillText(d,gx(i),H-4); });

  [['happy','#10b981'],['confused','#f59e0b'],['angry','#ef4444']].forEach(([key,col]) => {
    ctx.beginPath(); ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.lineJoin='round';
    data[key].forEach((v,i) => i===0 ? ctx.moveTo(gx(i),gy(v)) : ctx.lineTo(gx(i),gy(v)));
    ctx.stroke();
    data[key].forEach((v,i) => { ctx.beginPath(); ctx.arc(gx(i),gy(v),4,0,Math.PI*2); ctx.fillStyle=col; ctx.fill(); });
  });

  let lx=pL;
  [['Happy','#10b981'],['Confused','#f59e0b'],['Angry','#ef4444']].forEach(([lbl,col]) => {
    ctx.fillStyle=col; ctx.fillRect(lx,4,12,12);
    ctx.fillStyle='#6b7280'; ctx.font='11px Inter,sans-serif'; ctx.textAlign='left';
    ctx.fillText(lbl,lx+16,14); lx+=80;
  });
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2800);
}

function simulateEmotionChange() {
  const all = CLASSES.flatMap(c => c.students);
  const s = all[Math.floor(Math.random() * all.length)];
  const old = s.emotion;
  s.emotion = EMOTIONS[Math.floor(Math.random() * EMOTIONS.length)];
  if (s.emotion !== old) {
    const view = document.querySelector('.view.active');
    const vid = view ? view.id : '';
    if (vid === 'view-classDetail') {
      const cls = CLASSES.find(c => c.students.includes(s));
      if (cls && cls.id === currentClassId) renderStudentList(cls.students);
    }
    if (vid === 'view-overview') renderMoodBars();
  }
}
setInterval(simulateEmotionChange, 6000);

renderOverview();
