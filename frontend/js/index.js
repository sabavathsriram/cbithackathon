// Initialize AOS Animation
AOS.init({
    duration: 1000,
    once: true,
    easing: 'ease-out-cubic'
});

// Custom Cursor
const cursor = document.querySelector('.cursor');
const cursorFollower = document.querySelector('.cursor-follower');

document.addEventListener('mousemove', (e) => {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top = e.clientY + 'px';
    
    setTimeout(() => {
        cursorFollower.style.left = e.clientX + 'px';
        cursorFollower.style.top = e.clientY + 'px';
    }, 100);
});

// Navbar scroll effect
window.addEventListener('scroll', () => {
    const navbar = document.getElementById('mainNavbar');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            window.scrollTo({
                top: target.offsetTop - 80,
                behavior: 'smooth'
            });
        }
    });
});

// GSAP Animations
gsap.registerPlugin(ScrollTrigger);

// Floating particles animation
gsap.to(".particle-1", {
    y: -30,
    duration: 3,
    repeat: -1,
    yoyo: true,
    ease: "power1.inOut"
});

gsap.to(".particle-2", {
    y: -40,
    duration: 4,
    repeat: -1,
    yoyo: true,
    ease: "power1.inOut"
});

gsap.to(".particle-3", {
    y: -25,
    duration: 3.5,
    repeat: -1,
    yoyo: true,
    ease: "power1.inOut"
});

gsap.to(".particle-4", {
    y: -35,
    duration: 4.5,
    repeat: -1,
    yoyo: true,
    ease: "power1.inOut"
});

// Connection lines animation
gsap.fromTo(".line-1", 
    { width: 0 },
    { 
        width: "50%", 
        duration: 1.5, 
        scrollTrigger: {
            trigger: ".how-it-works",
            start: "top 80%"
        }
    }
);

gsap.fromTo(".line-2", 
    { width: 0 },
    { 
        width: "50%", 
        duration: 1.5, 
        delay: 0.5,
        scrollTrigger: {
            trigger: ".how-it-works",
            start: "top 80%"
        }
    }
);

// Body visualization glow effect
const bodyTooltips = document.querySelectorAll('.body-tooltip');
bodyTooltips.forEach(tooltip => {
    tooltip.addEventListener('mouseenter', () => {
        gsap.to(tooltip, {
            scale: 1.1,
            boxShadow: "0 8px 25px rgba(113, 201, 206, 0.8)",
            duration: 0.3
        });
    });
    
    tooltip.addEventListener('mouseleave', () => {
        gsap.to(tooltip, {
            scale: 1,
            boxShadow: "0 5px 15px rgba(0, 0, 0, 0.2)",
            duration: 0.3
        });
    });
});

// Initialize Lottie animations
document.addEventListener('DOMContentLoaded', () => {
    // Doctor animation
    const doctorAnimation = document.getElementById('doctor-animation');
    if (doctorAnimation) {
        doctorAnimation.innerHTML = `
            <lottie-player
                src="https://assets9.lottiefiles.com/packages/lf20_1pxqjqps.json"
                background="transparent"
                speed="1"
                style="width: 100%; height: 100%;"
                loop
                autoplay>
            </lottie-player>
        `;
    }
    
    // Body animation
    const bodyAnimation = document.getElementById('body-animation');
    if (bodyAnimation) {
        bodyAnimation.innerHTML = `
            <lottie-player
                src="https://assets2.lottiefiles.com/packages/lf20_DMgKk1.json"
                background="transparent"
                speed="1"
                style="width: 100%; height: 100%;"
                loop
                autoplay>
            </lottie-player>
        `;
    }
});

// Signup modal
const signupModal = new bootstrap.Modal(document.getElementById('signupModal'));
const signupButtons = document.querySelectorAll('a[href="#signup"]');
const signupForm = document.getElementById('signupForm');

signupButtons.forEach(button => {
    button.addEventListener('click', (e) => {
        e.preventDefault();
        signupModal.show();
    });
});

signupForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    // Get form values
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    // Here you would normally send the data to your backend
    console.log('Signup form submitted:', { name, email, password });
    
    // Show success message (in a real app, you'd handle this better)
    alert('Thank you for signing up! Welcome to SwasthyaSetu.');
    
    // Reset form and close modal
    signupForm.reset();
    signupModal.hide();
});

// Parallax effect for sections
gsap.utils.toArray("section").forEach(section => {
    gsap.to(section, {
        backgroundPosition: `50% ${innerHeight / 2}px`,
        ease: "none",
        scrollTrigger: {
            trigger: section,
            start: "top bottom",
            end: "bottom top",
            scrub: true
        }
    });
});

// Add hover effect to feature cards
const featureCards = document.querySelectorAll('.feature-card');
featureCards.forEach(card => {
    card.addEventListener('mouseenter', () => {
        gsap.to(card.querySelector('.feature-icon'), {
            rotation: 10,
            scale: 1.1,
            duration: 0.3
        });
    });
    
    card.addEventListener('mouseleave', () => {
        gsap.to(card.querySelector('.feature-icon'), {
            rotation: 0,
            scale: 1,
            duration: 0.3
        });
    });
});

// Add typing effect to hero subtitle
const heroSubtitle = document.querySelector('.hero-subtitle');
if (heroSubtitle) {
    const text = heroSubtitle.textContent;
    heroSubtitle.textContent = '';
    let i = 0;
    
    const typeWriter = () => {
        if (i < text.length) {
            heroSubtitle.textContent += text.charAt(i);
            i++;
            setTimeout(typeWriter, 50);
        }
    };
    
    // Start typing effect after page loads
    setTimeout(typeWriter, 1000);
}

// Add ripple effect to buttons
const buttons = document.querySelectorAll('.btn');
buttons.forEach(button => {
    button.addEventListener('click', function(e) {
        const ripple = document.createElement('span');
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;
        
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        ripple.classList.add('ripple');
        
        this.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
    });
});