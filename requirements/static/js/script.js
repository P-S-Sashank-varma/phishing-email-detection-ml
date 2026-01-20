// Predict Handler
async function handlePredict(event) {
    event.preventDefault();
    const emailContent = document.getElementById('email-content').value;
    const resultContainer = document.getElementById('result');

    if (!emailContent.trim()) {
        alert('Please enter email content');
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                email: emailContent
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        const prediction = data.prediction.toLowerCase();

        // Display result
        resultContainer.classList.remove('phishing', 'legitimate');
        
        if (prediction === 'phishing') {
            resultContainer.classList.add('phishing');
            document.getElementById('result-icon').textContent = '⚠️';
            document.getElementById('result-text').textContent = 'PHISHING DETECTED';
            document.getElementById('result-description').innerHTML = `
                <strong>This email appears to be a phishing attempt.</strong><br><br>
                This email contains characteristics commonly found in phishing attacks. 
                Be cautious and:<br>
                • Do not click any links<br>
                • Do not download attachments<br>
                • Do not reply with personal information<br>
                • Report it to your email provider
            `;
        } else {
            resultContainer.classList.add('legitimate');
            document.getElementById('result-icon').textContent = '✅';
            document.getElementById('result-text').textContent = 'LEGITIMATE EMAIL';
            document.getElementById('result-description').innerHTML = `
                <strong>This email appears to be legitimate.</strong><br><br>
                Our analysis suggests this email is safe. However, always remain cautious and 
                verify any requests through official channels before providing sensitive information.
            `;
        }

        resultContainer.style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
        alert('Error connecting to the server. Please try again.');
    }
}

// Contact Form Handler
function handleContactForm(event) {
    event.preventDefault();
    const messageBox = document.getElementById('contact-message');
    
    messageBox.textContent = 'Thank you! Your message has been sent. We will get back to you soon.';
    messageBox.classList.remove('error');
    messageBox.classList.add('success');
    messageBox.style.display = 'block';

    event.target.reset();

    setTimeout(() => {
        messageBox.style.display = 'none';
    }, 5000);
}

// Set active nav link
document.addEventListener('DOMContentLoaded', function() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
});
