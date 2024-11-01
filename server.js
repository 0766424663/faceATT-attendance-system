const express = require('express');
const app = express();
const PORT = 3000;

app.get('/', (req, res) => {
    res.send("Server is running!");
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
app.use(express.json());

app.post('/register-face', (req, res) => {
    // Code to save face data
    res.send("Face registered in back-end!");
});

app.post('/mark-attendance', (req, res) => {
    // Code to log attendance
    res.send("Attendance logged!");
});
