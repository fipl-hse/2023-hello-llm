'use strict'

let inferSampleFromForm = async (question, result) => {
    let response = await fetch('/infer', {
        method: 'POST',
        headers: {
            'Content-type': 'application/json'
        },
        body: JSON.stringify({
            question: question.value,
        })
    })
    if (response.ok) {
        let infer_res = await response.json();
        result.innerHTML = '';
        result.appendChild(document.createTextNode(infer_res['infer']))
    } else {
        result.appendChild(document.createTextNode('Something went wrong: ' + response.status))
    }
}

let disableButton = (btn, question) => {
    btn.disabled = !(question.value);
}

window.onload = function() {
    const btn = document.getElementById('btn_submit');
    const question = document.getElementById('question');
    const showResults = document.getElementById('result');

    disableButton(
        btn,
        question,
    );

    question.addEventListener('change', () => {
        disableButton(btn, question)
    });

    btn.addEventListener('click', () => {
        inferSampleFromForm(question, showResults)
    })
}

