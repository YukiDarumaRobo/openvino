function sendCommand(command) {
    console.log({action: 'sendCommand', command: command})
    const form = new FormData()
    form.append('command', command)
    fetch('/api/command/', {
        method: 'POST',
        body: form
    }).then((json) => {
        console.log({action: 'sendCommand', json: json})
    }, 'json')
}