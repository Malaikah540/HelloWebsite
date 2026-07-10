const express = require('express');
const http    = require('http');
const { Server } = require('socket.io');
const path    = require('path');

const app    = express();
const server = http.createServer(app);
const io     = new Server(server);

app.use(express.json());
app.use(express.static(path.join(__dirname)));

/* ===== IN-MEMORY DATA ===== */
const NAMES = [
  'Ahmed Raza','Fatima Khan','Usman Tariq','Ayesha Mir','Bilal Shah',
  'Zara Iqbal','Hassan Ali','Maryam Butt','Omar Cheema','Hina Siddiqui',
  'Saad Malik','Noor Baig','Kamran Javed','Sara Naz','Imran Qureshi',
  'Layla Rehman','Tariq Mehmood','Amna Arif','Asad Riaz','Dua Farooq',
];
const AVATAR_COLORS = ['#10b981','#3b82f6','#8b5cf6','#f59e0b','#ef4444','#06b6d4','#ec4899'];
const EMOTIONS      = ['happy','happy','happy','confused','confused','angry'];

function genStudents(classId, count) {
  const shuffled = [...NAMES].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count).map((name, i) => ({
    id:          `S${classId}${String(i + 1).padStart(2, '0')}`,
    name,
    emotion:     EMOTIONS[Math.floor(Math.random() * EMOTIONS.length)],
    avatarColor: AVATAR_COLORS[i % AVATAR_COLORS.length],
    initials:    name.split(' ').map(w => w[0]).join(''),
    note:        '',
  }));
}

const CLASSES = [
  { id: 1, subject: 'Robotics Fundamentals',  grade: 'Grade 7A', days: 'Mon / Wed / Fri', time: '08:00', end: '09:00', room: 'Lab 3',  students: genStudents(1, 18) },
  { id: 2, subject: 'AI & Coding',            grade: 'Grade 8B', days: 'Tue / Thu',       time: '09:30', end: '10:30', room: 'Lab 1',  students: genStudents(2, 14) },
  { id: 3, subject: 'Electronics Workshop',   grade: 'Grade 6C', days: 'Mon / Wed',       time: '11:00', end: '12:00', room: 'Lab 2',  students: genStudents(3, 16) },
  { id: 4, subject: 'Computational Thinking', grade: 'Grade 9A', days: 'Tue / Fri',       time: '13:00', end: '14:00', room: 'Room 5', students: genStudents(4, 10) },
];

const interventionLog = []; // { studentId, intervention, timestamp }

/* ===== REST API ===== */

// GET /api/classes  — list all classes (no students)
app.get('/api/classes', (req, res) => {
  res.json(CLASSES.map(({ students, ...c }) => c));
});

// GET /api/classes/:id/students
app.get('/api/classes/:id/students', (req, res) => {
  const cls = CLASSES.find(c => c.id === parseInt(req.params.id));
  if (!cls) return res.status(404).json({ error: 'Class not found' });
  res.json(cls.students);
});

// GET /api/students  — all students across all classes
app.get('/api/students', (req, res) => {
  const all = CLASSES.flatMap(c => c.students.map(s => ({ ...s, classId: c.id, classLabel: `${c.subject} - ${c.grade}` })));
  res.json(all);
});

// PATCH /api/students/:id/emotion
app.patch('/api/students/:id/emotion', (req, res) => {
  const { emotion } = req.body;
  if (!['happy', 'confused', 'angry'].includes(emotion))
    return res.status(400).json({ error: 'Invalid emotion' });

  for (const cls of CLASSES) {
    const s = cls.students.find(s => s.id === req.params.id);
    if (s) {
      s.emotion = emotion;
      io.emit('emotionUpdate', { studentId: s.id, emotion, classId: cls.id });
      return res.json({ ok: true });
    }
  }
  res.status(404).json({ error: 'Student not found' });
});

// PATCH /api/students/:id/note
app.patch('/api/students/:id/note', (req, res) => {
  const { note } = req.body;
  for (const cls of CLASSES) {
    const s = cls.students.find(s => s.id === req.params.id);
    if (s) { s.note = note; return res.json({ ok: true }); }
  }
  res.status(404).json({ error: 'Student not found' });
});

// POST /api/interventions
app.post('/api/interventions', (req, res) => {
  const { studentId, intervention } = req.body;
  if (!studentId || !intervention)
    return res.status(400).json({ error: 'studentId and intervention required' });

  const entry = { studentId, intervention, timestamp: new Date().toISOString() };
  interventionLog.push(entry);
  console.log(`[Intervention] ${studentId} -> ${intervention}`);
  io.emit('interventionApplied', entry);
  res.json({ ok: true });
});

// GET /api/interventions  — view log
app.get('/api/interventions', (req, res) => {
  res.json(interventionLog.slice(-50)); // last 50
});

/* ===== SOCKET.IO ===== */
io.on('connection', socket => {
  console.log(`[Socket] Client connected: ${socket.id}`);

  // Send current snapshot on connect
  socket.emit('snapshot', {
    classes: CLASSES.map(({ students, ...c }) => ({ ...c, studentCount: students.length })),
    emotions: CLASSES.flatMap(c => c.students.map(s => ({ studentId: s.id, emotion: s.emotion, classId: c.id }))),
  });

  socket.on('disconnect', () => console.log(`[Socket] Client disconnected: ${socket.id}`));
});

/* ===== SIMULATE REAL-TIME EMOTION CHANGES ===== */
setInterval(() => {
  const allStudents = CLASSES.flatMap(c => c.students.map(s => ({ ...s, classId: c.id })));
  const pick        = allStudents[Math.floor(Math.random() * allStudents.length)];
  const newEmotion  = EMOTIONS[Math.floor(Math.random() * EMOTIONS.length)];
  if (newEmotion !== pick.emotion) {
    for (const cls of CLASSES) {
      const s = cls.students.find(s => s.id === pick.id);
      if (s) {
        s.emotion = newEmotion;
        io.emit('emotionUpdate', { studentId: s.id, emotion: newEmotion, classId: cls.id });
        break;
      }
    }
  }
}, 6000);

/* ===== START SERVER ===== */
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`\nGrowBots Teacher Dashboard`);
  console.log(`Running on http://localhost:${PORT}\n`);
});
