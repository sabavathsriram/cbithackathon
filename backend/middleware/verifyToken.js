const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

const verifyToken = (req, res, next) => {
    // --- DEBUGGING LOGS ---
    console.log('--- VERIFY TOKEN MIDDLEWARE ---');
    console.log('Request Path:', req.path);
    console.log('Request Cookies:', req.cookies);
    console.log('Authorization Header:', req.headers.authorization);
    console.log('-------------------------------');

    // Check for token in Authorization header first, then in cookies
    const token = req.headers.authorization || req.cookies.token;

    if (!token) {
        console.log('VERIFICATION FAILED: No token found.');
        return res.status(401).json({ message: 'No token provided, authorization denied' });
    }

    // Remove "Bearer " prefix if present
    const tokenWithoutBearer = token.startsWith('Bearer ') ? token.slice(7) : token;

    jwt.verify(tokenWithoutBearer, JWT_SECRET, (err, decoded) => {
        if (err) {
            console.log('VERIFICATION FAILED: Invalid token.', err.message);
            return res.status(401).json({ message: 'Token is not valid' });
        }
        console.log('VERIFICATION SUCCESS!');
        req.user = decoded;
        next();
    });
};

module.exports = { verifyToken };