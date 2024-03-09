document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('submit').addEventListener('click', async function() {
        const question = document.getElementById('question').value;
            const response = await fetch('/infer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });
            const data = await response.json();
            const predictedLang = data.infer;


            document.getElementById('output').innerHTML = predictedLang;

    });
});