// --- LOAD ENVIRONMENT VARIABLES FIRST ---
// This ensures all subsequent files have access to process.env variables.
const dotenv = require('dotenv');
dotenv.config();
// ---------------------------------------

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const cookieParser = require('cookie-parser');

// Import routes
const userRoutes = require('./routes/userRoutes');

const app = express();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors({
    origin: 'http://localhost:5000', // Ensure this matches your frontend's origin
    credentials: true,
}));
app.use(cookieParser());

// Serve static files from the frontend directory
// The path is '../frontend' because server.js is in the 'backend' folder.
app.use(express.static(path.join(__dirname, '../frontend')));

// MongoDB Connection
mongoose.connect(process.env.MONGO_URI, {
        // useNewUrlParser: true,
        // useUnifiedTopology: true,
    })
    .then(() => console.log('MongoDB connected'))
    .catch(err => console.error('MongoDB connection error:', err));

// Use routes
app.use('/', userRoutes);

// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));