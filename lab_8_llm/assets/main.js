'use strict'

let inferSampleFromForm = async (premise, result) => {
    let response = await fetch('/infer', {
        method: 'POST',
        headers: {
            'Content-type': 'application/json'
        },
        body: JSON.stringify({
            question: premise.value,
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

let disableButton = (btn, premise) => {
    btn.disabled = !(premise.value);
}

window.onload = function() {
    const btn = document.getElementById('btn_submit');
    const premise = document.getElementById('premise');
    const showResults = document.getElementById('result');

    disableButton(
        btn,
        premise,
    );

    premise.addEventListener('change', () => {
        disableButton(btn, premise)
    });

    btn.addEventListener('click', () => {
        inferSampleFromForm(premise, showResults)
    })
}

