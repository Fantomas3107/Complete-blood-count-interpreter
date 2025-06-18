document.getElementById('bloodAnalysisForm').addEventListener('submit', function(e) {
    const submitButton = this.querySelector('button[type="submit"]');
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Анализируем...';
    submitButton.disabled = true;
});