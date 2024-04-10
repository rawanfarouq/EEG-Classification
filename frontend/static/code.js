document.addEventListener('DOMContentLoaded', (event) => {
    document.querySelector('form').onsubmit = function(event) {
        console.log('Form submission intercepted.');
        event.preventDefault(); // Continue to prevent the traditional form submission
        var formData = new FormData(this);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        }).then(response => response.text()) // Expect text response now
        .then(html => {
            // Find the element where you want to display the alert
            var alertBox = document.getElementById('alertBox');
            // Set the innerHTML of that element to the response HTML
            alertBox.innerHTML = html;
        }).catch(error => {
            console.error('Error:', error);
        });
    };
});