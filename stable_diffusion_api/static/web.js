document.getElementById('generate-button').addEventListener('click', function() {
    const blueprintName = document.getElementById('blueprint-select').value;

    fetch('/generate-image/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ blueprint_name: blueprintName }),
    })
    .then(response => response.json())
    .then(data => {
        const imageUrl = data.image_url;
        
        // Display the generated image
        const imgElement = document.getElementById('generated-image');
        imgElement.src = imageUrl;
        
        imgElement.classList.add('fade-in');
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
