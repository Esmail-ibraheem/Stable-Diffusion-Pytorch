const hardwareOptions = ["CPU", "CUDA", "MPS"];
        const hardwareSelect = document.getElementById('hardware-select');
        const projectCreationSection = document.getElementById('project-creation');
        const imageGenerationSection = document.getElementById('image-generation');
        const gallerySection = document.getElementById('gallery');
        const pageTitle = document.getElementById('page-title');

        hardwareOptions.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option;
            opt.textContent = option;
            hardwareSelect.appendChild(opt);
        });

        function createProject() {
            const projectName = document.getElementById('project-name').value;
            const projectDescription = document.getElementById('project-description').value;
            const selectedHardware = hardwareSelect.value;

            if (!projectName || !selectedHardware) {
                alert('Please fill in the project name and select hardware.');
                return;
            }

            const button = event.target;
            button.textContent = "Creating...";
            button.disabled = true;

            setTimeout(() => {
                button.textContent = "Create Project";
                button.disabled = false;

                projectCreationSection.classList.remove('active');
                imageGenerationSection.classList.add('active');
                gallerySection.classList.add('active');
                pageTitle.textContent = "Generate Image";
            }, 1000); // Simulate a delay for creating the project
        }

        async function generateImage() {
            const prompt = document.getElementById('prompt-input').value;
            if (!prompt) {
                alert('Please enter a prompt.');
                return;
            }

            // Simulate a loading state on the button
            const button = event.target;
            button.textContent = "Generating...";
            button.disabled = true;

            try {
                const response = await fetch('/generate/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ prompt: prompt })
                });

                if (!response.ok) {
                    throw new Error('Failed to generate image');
                }

                const data = await response.json();
                const imageContainer = document.getElementById('gallery');

                const img = document.createElement('img');
                img.src = `data:image/png;base64,${data.image}`;
                img.alt = 'Generated Image';

                const downloadLink = document.createElement('a');
                downloadLink.href = img.src;
                downloadLink.download = 'generated_image.png';
                downloadLink.classList.add('download-button');
                downloadLink.textContent = 'Download Image';

                const imgContainer = document.createElement('div');
                imgContainer.classList.add('image-container');
                imgContainer.appendChild(img);
                imgContainer.appendChild(downloadLink);

                imageContainer.appendChild(imgContainer);

                button.textContent = "Generate";
                button.disabled = false;
            } catch (error) {
                console.error('Error generating image:', error);
                alert('An error occurred while generating the image.');
                button.textContent = "Generate";
                button.disabled = false;
            }
        }
