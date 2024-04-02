console.log('JavaScript file is loaded.');
document.addEventListener('DOMContentLoaded', (event) => {
    document.querySelector('form').onsubmit = function(event) {
        console.log('Form submission intercepted.');
        event.preventDefault(); // Prevent the form from submitting the traditional way
        var formData = new FormData(this);
        fetch('/', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
        .then(data => {
            console.log('Response received:', data);
            alert(data.message); // Alert the message from the server
        }).catch(error => {
            console.error('Error:', error);
        });
    };
});