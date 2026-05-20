const canvas = document.getElementById('annotatorCanvas');
const ctx = canvas.getContext('2d');
const imageSelector = document.getElementById('imageSelector');
const classNamesInput = document.getElementById('classNames');
const currentLabelInput = document.getElementById('currentLabel');
const boxesList = document.getElementById('boxesList');
const classChips = document.getElementById('classChips');
const selectedClassName = document.getElementById('selectedClassName');
const newClassInput = document.getElementById('newClassInput');
const uploadFileInput = document.getElementById('uploadFile');

let loadedImage = new Image();
let currentImageName = '';
let boxes = [];
let previewBox = null;
let isDrawing = false;
let startPoint = null;
let scaleX = 1;
let scaleY = 1;
let classes = [];
let imageLoaded = false;

function normalizeClassName(name) {
    return (name || '').trim().replace(/\s+/g, '_');
}

function syncClassInputs() {
    if (!classes.length) {
        classes = ['object'];
    }
    classes = [...new Set(classes.map(normalizeClassName).filter(Boolean))];
    classNamesInput.value = classes.join(',');
    if (!classes.includes(currentLabelInput.value)) {
        currentLabelInput.value = classes[0];
    }
    selectedClassName.textContent = currentLabelInput.value || 'object';
}

function renderClassChips() {
    syncClassInputs();
    classChips.innerHTML = '';
    classes.forEach((label) => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = `class-chip ${label === currentLabelInput.value ? 'active' : ''}`;
        chip.onclick = () => selectClass(label);

        const text = document.createElement('span');
        text.textContent = label;
        chip.appendChild(text);

        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'class-chip-remove';
        remove.textContent = '×';
        remove.title = `Видалити клас ${label}`;
        remove.onclick = (event) => {
            event.stopPropagation();
            removeClass(label);
        };
        chip.appendChild(remove);
        classChips.appendChild(chip);
    });
}

function initializeClasses() {
    const initial = classNamesInput.value.split(',').map(normalizeClassName).filter(Boolean);
    classes = [...new Set(initial.length ? initial : ['person', 'car', 'bicycle'])];
    currentLabelInput.value = normalizeClassName(currentLabelInput.value) || classes[0];
    renderClassChips();
}

function selectClass(label) {
    currentLabelInput.value = label;
    renderClassChips();
}

function addClassFromInput() {
    const value = normalizeClassName(newClassInput.value);
    if (!value) return;
    if (!classes.includes(value)) {
        classes.push(value);
    }
    currentLabelInput.value = value;
    newClassInput.value = '';
    renderClassChips();
}

function removeClass(label) {
    if (classes.length === 1) {
        alert('Має залишитися хоча б один клас');
        return;
    }
    const hasBoxes = boxes.some((box) => box.label === label);
    if (hasBoxes) {
        const confirmed = confirm(`Клас ${label} використовується в розмітці. Видалити його зі списку? Існуючі рамки з цим класом залишаться.`);
        if (!confirmed) return;
    }
    classes = classes.filter((item) => item !== label);
    if (currentLabelInput.value === label) {
        currentLabelInput.value = classes[0];
    }
    renderClassChips();
}

function getClasses() {
    syncClassInputs();
    return classes;
}

function getCanvasClientWidth() {
    const wrap = canvas.parentElement;
    const wrapWidth = wrap ? wrap.clientWidth - 24 : window.innerWidth - 180;
    return Math.max(320, wrapWidth);
}

function setCanvasSize(img) {
    const maxWidth = Math.min(getCanvasClientWidth(), 1240);
    const maxHeight = Math.max(320, window.innerHeight - 260);
    const ratio = img.width / img.height || 1;

    let width = Math.min(maxWidth, img.width);
    let height = width / ratio;

    if (height > maxHeight) {
        height = maxHeight;
        width = height * ratio;
    }

    canvas.width = Math.max(1, Math.round(width));
    canvas.height = Math.max(1, Math.round(height));
    canvas.style.width = `${canvas.width}px`;
    canvas.style.height = `${canvas.height}px`;
    scaleX = img.width / canvas.width;
    scaleY = img.height / canvas.height;
}

function drawBoxOnCanvas(box, index) {
    const x = box.x1 / scaleX;
    const y = box.y1 / scaleY;
    const w = (box.x2 - box.x1) / scaleX;
    const h = (box.y2 - box.y1) / scaleY;

    ctx.strokeStyle = '#0ea5a4';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x, y, w, h);

    const title = `${index + 1}. ${box.label}`;
    const titleWidth = Math.max(92, ctx.measureText(title).width + 18);
    const titleY = Math.max(0, y - 26);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(x, titleY, titleWidth, 24);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(title, x + 8, titleY + 16);
}

function renderCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imageLoaded) {
        ctx.drawImage(loadedImage, 0, 0, canvas.width, canvas.height);
    }
    ctx.font = '14px Inter, sans-serif';

    boxes.forEach((box, index) => drawBoxOnCanvas(box, index));

    if (previewBox) {
        ctx.strokeStyle = '#f97316';
        ctx.lineWidth = 2;
        ctx.strokeRect(previewBox.x, previewBox.y, previewBox.w, previewBox.h);
    }
    renderBoxesList();
}

function renderBoxesList() {
    boxesList.innerHTML = '';
    if (!boxes.length) {
        boxesList.innerHTML = '<div class="empty-state">Ще немає розмітки. Обери клас і виділи перший об’єкт на полотні.</div>';
        return;
    }
    boxes.forEach((box, index) => {
        const row = document.createElement('div');
        row.className = 'list-item';
        row.innerHTML = `<div class="list-main"><strong>${index + 1}. ${box.label}</strong><span>[${box.x1}, ${box.y1}, ${box.x2}, ${box.y2}]</span></div>`;
        const button = document.createElement('button');
        button.className = 'secondary-btn compact-btn';
        button.textContent = 'Видалити';
        button.onclick = () => {
            boxes.splice(index, 1);
            renderCanvas();
        };
        row.appendChild(button);
        boxesList.appendChild(row);
    });
}

function resetCurrentCanvas() {
    boxes = [];
    previewBox = null;
    imageLoaded = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    renderBoxesList();
}

function refreshStats(stats) {
    if (!stats) return;
    document.getElementById('trainImages').textContent = stats.train_images ?? 0;
    document.getElementById('valImages').textContent = stats.val_images ?? 0;
    document.getElementById('trainLabels').textContent = stats.train_labels ?? 0;
    document.getElementById('valLabels').textContent = stats.val_labels ?? 0;
}

async function fetchAnnotation(imageName) {
    const response = await fetch(`/api/annotator/annotation?image_name=${encodeURIComponent(imageName)}`);
    const data = await response.json();
    if (!data.success) {
        throw new Error(data.error || 'Не вдалося завантажити розмітку');
    }
    return data.annotation;
}

async function loadSelectedImage() {
    const selected = imageSelector.value;
    if (!selected) {
        currentImageName = '';
        loadedImage = new Image();
        resetCurrentCanvas();
        return;
    }

    currentImageName = selected;
    const annotation = await fetchAnnotation(selected).catch(() => null);
    if (annotation?.classes?.length) {
        classes = [...new Set(annotation.classes.map(normalizeClassName).filter(Boolean))];
        if (annotation.boxes?.length) {
            const usedLabels = annotation.boxes.map((box) => normalizeClassName(box.label)).filter(Boolean);
            classes = [...new Set([...classes, ...usedLabels])];
        }
        if (classes.length) {
            currentLabelInput.value = classes[0];
        }
        renderClassChips();
    }

    loadedImage = new Image();
    loadedImage.onload = () => {
        setCanvasSize(loadedImage);
        imageLoaded = true;
        boxes = (annotation?.boxes || []).map((box) => ({
            label: normalizeClassName(box.label),
            x1: Number(box.x1),
            y1: Number(box.y1),
            x2: Number(box.x2),
            y2: Number(box.y2),
        }));
        renderCanvas();
    };
    loadedImage.onerror = () => {
        resetCurrentCanvas();
        alert('Не вдалося завантажити зображення');
    };
    loadedImage.src = `/storage/uploads/${selected}?t=${Date.now()}`;
}

function canvasPointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) * canvas.width) / rect.width;
    const y = ((event.clientY - rect.top) * canvas.height) / rect.height;
    return { x: Math.max(0, Math.min(canvas.width, x)), y: Math.max(0, Math.min(canvas.height, y)) };
}

imageSelector?.addEventListener('change', () => {
    loadSelectedImage().catch((error) => alert(error.message || String(error)));
});

newClassInput?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault();
        addClassFromInput();
    }
});

window.addEventListener('resize', () => {
    if (imageLoaded) {
        setCanvasSize(loadedImage);
        renderCanvas();
    }
});

document.getElementById('uploadForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = uploadFileInput.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/api/annotator/upload', { method: 'POST', body: formData });
    const data = await response.json();
    if (!data.success) {
        alert(data.error || 'Помилка завантаження');
        return;
    }

    let option = [...imageSelector.options].find((item) => item.value === data.file_name);
    if (!option) {
        option = document.createElement('option');
        option.value = data.file_name;
        option.textContent = data.file_name;
        imageSelector.prepend(option);
    }
    imageSelector.value = data.file_name;
    uploadFileInput.value = '';
    await loadSelectedImage();
});

canvas?.addEventListener('mousedown', (event) => {
    if (!imageLoaded) return;
    startPoint = canvasPointFromEvent(event);
    isDrawing = true;
});

canvas?.addEventListener('mousemove', (event) => {
    if (!isDrawing || !startPoint) return;
    const point = canvasPointFromEvent(event);
    previewBox = {
        x: Math.min(startPoint.x, point.x),
        y: Math.min(startPoint.y, point.y),
        w: Math.abs(point.x - startPoint.x),
        h: Math.abs(point.y - startPoint.y),
    };
    renderCanvas();
});

canvas?.addEventListener('mouseup', (event) => {
    if (!isDrawing || !startPoint) return;
    isDrawing = false;
    const point = canvasPointFromEvent(event);
    const x1 = Math.round(Math.min(startPoint.x, point.x) * scaleX);
    const y1 = Math.round(Math.min(startPoint.y, point.y) * scaleY);
    const x2 = Math.round(Math.max(startPoint.x, point.x) * scaleX);
    const y2 = Math.round(Math.max(startPoint.y, point.y) * scaleY);

    if (Math.abs(x2 - x1) > 8 && Math.abs(y2 - y1) > 8) {
        boxes.push({ label: currentLabelInput.value.trim() || 'object', x1, y1, x2, y2 });
    }
    startPoint = null;
    previewBox = null;
    renderCanvas();
});

canvas?.addEventListener('mouseleave', () => {
    if (!isDrawing) return;
    isDrawing = false;
    startPoint = null;
    previewBox = null;
    renderCanvas();
});

function clearBoxes() {
    boxes = [];
    previewBox = null;
    renderCanvas();
}

async function saveAnnotation() {
    if (!currentImageName || !imageLoaded) {
        alert('Спочатку обери зображення');
        return;
    }
    const payload = {
        image_name: currentImageName,
        image_width: loadedImage.naturalWidth || loadedImage.width,
        image_height: loadedImage.naturalHeight || loadedImage.height,
        classes: getClasses(),
        boxes,
    };
    const response = await fetch('/api/annotator/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!data.success) {
        alert(data.error || 'Помилка збереження');
        return;
    }
    refreshStats(data.stats);
    await loadSelectedImage();
    alert(`Розмітку збережено. Split: ${data.record?.split || 'train'}`);
}

async function deleteSelectedImage() {
    const imageName = imageSelector.value;
    if (!imageName) {
        alert('Спочатку обери файл');
        return;
    }
    const confirmed = confirm(`Видалити файл ${imageName} разом із його labels?`);
    if (!confirmed) return;

    const response = await fetch('/api/annotator/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_name: imageName })
    });
    const data = await response.json();
    if (!data.success) {
        alert(data.error || 'Помилка видалення');
        return;
    }

    const option = [...imageSelector.options].find((item) => item.value === imageName);
    if (option) option.remove();
    refreshStats(data.stats);
    if (imageSelector.options.length) {
        imageSelector.selectedIndex = 0;
        await loadSelectedImage();
    } else {
        currentImageName = '';
        loadedImage = new Image();
        resetCurrentCanvas();
    }
}

window.clearBoxes = clearBoxes;
window.saveAnnotation = saveAnnotation;
window.deleteSelectedImage = deleteSelectedImage;
window.addClassFromInput = addClassFromInput;

initializeClasses();
if (imageSelector?.value) {
    loadSelectedImage().catch((error) => console.error(error));
} else {
    renderBoxesList();
}
