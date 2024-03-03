let inferSample = async (premise, hypothesis, result) => {
    const response = await fetch('/infer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            question: premise.value,
            hypothesis: hypothesis.value
        })
    });
    const data = await response.json();
    result.innerHTML = '';
    result.appendChild(document.createTextNode(data['infer'].toUpperCase()));
};

window.onload = function() {
    let btn = document.getElementById('submit');
    let premise = document.getElementById('premise');
    let hypothesis = document.getElementById('hype');
    let prediction = document.getElementById('pred');

    btn.addEventListener('click', () => {
        inferSample(premise, hypothesis, prediction)
    })
}
