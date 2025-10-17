const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { User, Patient, Doctor } = require('../model/user');
const path = require('path');
const { verifyToken } = require('../middleware/verifyToken'); // Import the middleware

const router = express.Router();

// dotenv.config() is no longer needed here because it's called in server.js
// before this file is imported.

// JWT Secret Key (will now correctly read from process.env)
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// Page Routes - Serve HTML files
router.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../../frontend/views/signin.html'));
});

router.get('/login', (req, res) => {
    res.sendFile(path.join(__dirname, '../../frontend/views/signin.html'));
});

router.get('/signup', (req, res) => {
    res.sendFile(path.join(__dirname, '../../frontend/views/signup.html'));
});

// This single dashboard route handles both patient and doctor redirection
router.get('/dashboard', verifyToken, async(req, res) => {
    try {
        const user = await User.findById(req.user.id).select('-password');
        if (!user) {
            return res.redirect('/login');
        }
        if (user.role === 'patient') {
            res.sendFile(path.join(__dirname, '../../frontend/views/dashboard.html'));
        } else if (user.role === 'doctor') {
            res.sendFile(path.join(__dirname, '../../frontend/views/doctor_dashboard.html'));
        } else {
            return res.status(400).json({ message: 'Invalid user role' });
        }
    } catch (error) {
        console.error('Dashboard error:', error);
        res.redirect('/login');
    }
});

// API Routes
// User Signup
router.post('/api/auth/signup', async(req, res) => {
    const { fullName, email, password, role, dob, gender, specialization, license, experience } = req.body;

    try {
        // ... (validation logic remains the same)
        if (!fullName || !email || !password || !role) {
            return res.status(400).json({ message: 'All required fields must be provided' });
        }
        if (password.length < 8) {
            return res.status(400).json({ message: 'Password must be at least 8 characters long' });
        }
        if (role === 'patient' && (!dob || !gender)) {
            return res.status(400).json({ message: 'Date of birth and gender are required for patients' });
        }
        if (role === 'doctor' && (!specialization || !license || !experience)) {
            return res.status(400).json({ message: 'Specialization, license, and experience are required for doctors' });
        }
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: 'Email already in use' });
        }
        const hashedPassword = await bcrypt.hash(password, 10);
        const user = new User({ fullName, email, password: hashedPassword, role });
        await user.save();
        if (role === 'patient') {
            const patient = new Patient({ userId: user._id, dob: new Date(dob), gender });
            await patient.save();
        } else if (role === 'doctor') {
            const doctor = new Doctor({ userId: user._id, specialization, license, experience });
            await doctor.save();
        }
        const token = jwt.sign({ id: user._id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '1h' });

        // Set token in cookie with correct path and sameSite attributes
        res.cookie('token', token, {
            httpOnly: true,
            maxAge: 3600000,
            path: '/', // Available on all paths
            sameSite: 'Lax' // Recommended for security and compatibility
        });

        res.status(201).json({
            message: 'Sign up successful!',
            user: { id: user._id, fullName: user.fullName, email: user.email, role: user.role },
        });
    } catch (error) {
        console.error('Signup error:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// User Login
router.post('/api/auth/login', async(req, res) => {
    const { email, password } = req.body;

    try {
        // ... (validation logic remains the same)
        if (!email || !password) {
            return res.status(400).json({ message: 'Email and password are required' });
        }
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }
        const token = jwt.sign({ id: user._id, email: user.email, role: user.role }, JWT_SECRET, { expiresIn: '1h' });

        // Set token in cookie with correct path and sameSite attributes
        res.cookie('token', token, {
            httpOnly: true,
            maxAge: 3600000,
            path: '/', // Available on all paths
            sameSite: 'Lax' // Recommended for security and compatibility
        });

        res.status(200).json({
            message: 'Login successful',
            user: { id: user._id, fullName: user.fullName, email: user.email, role: user.role },
        });
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// User Logout
router.post('/api/auth/logout', (req, res) => {
    res.clearCookie('token');
    res.status(200).json({ message: 'Logout successful' });
});

// Get current user info
router.get('/api/auth/user', verifyToken, async(req, res) => {
    try {
        const user = await User.findById(req.user.id).select('-password');
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }
        let profile = null;
        if (user.role === 'patient') {
            profile = await Patient.findOne({ userId: user._id });
        } else if (user.role === 'doctor') {
            profile = await Doctor.findOne({ userId: user._id });
        }
        res.status(200).json({
            user: { id: user._id, fullName: user.fullName, email: user.email, role: user.role, profile },
        });
    } catch (error) {
        console.error('Get user error:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

module.exports = router;