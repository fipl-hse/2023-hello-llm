document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('submit').addEventListener('click', async function() {
        const question = document.getElementById('question').value;
        if (question.trim() !== '') {
            const response = await fetch('/infer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });
            const data = await response.json();
            const prompt = `<span class="prompt">${question}</span> `;
            const generatedText = data.infer;
            function typeWriter(text, elementId, delay = 50) {
                let i = 1;
                for (let letter of text) {
                  setTimeout(function () {
                      document.getElementById(elementId).innerHTML += letter;
                      console.log(letter)
                  }, delay * i)
                    i++;
                }
            }
            document.getElementById('output').innerHTML = '';
            document.getElementById('output').innerHTML = prompt;
            typeWriter(generatedText.substring(question.length), 'output', 25);
        } else {
            document.getElementById('output').textContent = 'Please enter some text.';
        }
    });
});
