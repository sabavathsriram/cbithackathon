const mongoose = require('mongoose');

// User Schema (Base Schema)
const userSchema = new mongoose.Schema({
    fullName: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    role: { type: String, enum: ['patient', 'doctor'], required: true },
    isVerified: { type: Boolean, default: false },
    createdAt: { type: Date, default: Date.now },
});

// Patient Schema
const patientSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    dob: { type: Date, required: true },
    gender: { type: String, enum: ['male', 'female', 'other'], required: true },
});

// Doctor Schema
const doctorSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    specialization: {
        type: String,
        enum: [
            'cardiology', 'dermatology', 'endocrinology', 'gastroenterology',
            'neurology', 'oncology', 'pediatrics', 'psychiatry', 'radiology',
            'surgery', 'other'
        ],
        required: true
    },
    license: { type: String, required: true },
    experience: { type: Number, required: true, min: 0 },
});

const User = mongoose.model('User', userSchema);
const Patient = mongoose.model('Patient', patientSchema);
const Doctor = mongoose.model('Doctor', doctorSchema);

module.exports = { User, Patient, Doctor };