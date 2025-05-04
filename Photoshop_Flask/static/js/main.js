document.addEventListener('DOMContentLoaded', function() {
    // 显示与隐藏进度条
    function showProgress() {
        document.getElementById('progress-indicator').style.display = '';
    }
    function hideProgress() {
        document.getElementById('progress-indicator').style.display = 'none';
    }

    // 菜单切换
    document.querySelectorAll('.menu-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.menu-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            document.querySelectorAll('.feature-panel').forEach(panel => panel.style.display = 'none');
            document.getElementById(this.dataset.target + '-panel').style.display = '';
        });
    });

    // 图片上传（除了拼接模块）
    let uploadedFilenames = [];
    const uploadInput = document.getElementById('upload-input');
    if(uploadInput){
        uploadInput.addEventListener('change', function(){
            let files = this.files;
            let formData = new FormData();
            for(let i=0; i<files.length; i++){
                formData.append('file', files[i]);
            }
            showProgress();
            fetch('/upload', {method:'POST', body:formData})
                .then(r=>r.json())
                .then(res=>{
                    hideProgress();
                    if(res.filenames && res.filenames.length>0){
                        uploadedFilenames = res.filenames;
                        let imgUrl = '/uploads/' + uploadedFilenames[0];
                        document.getElementById('original-img-processing').src = imgUrl;
                        document.getElementById('original-img-segmentation').src = imgUrl;
                        document.getElementById('original-img-restoration').src = imgUrl;
                        document.getElementById('download-btn').disabled = true;
                    }
                })
                .catch(()=>{
                    hideProgress();
                    alert('上传失败，请重试');
                });
        });
    }

     // 图像处理参数联动
    const processingType = document.getElementById('processing-type');
    const processingParams = document.getElementById('processing-params');
    function updateProcessingParams(){
        let type = processingType.value;
        let html = '';
        if(type==='gray'){
            html = `<select id="gray-method">
                        <option value="linear">线性拉伸</option>
                        <option value="log">对数变换</option>
                        <option value="exp">指数变换</option>
                    </select>`;
        }else if(type==='binary'){
            html = `<label>阈值: <input type="number" id="binary-threshold" value="128" min="0" max="255" style="width:60px"></label>`;
        }else if(type==='smooth'){
            html = `<select id="smooth-method">
                        <option value="mean">领域平均法</option>
                        <option value="gaussian">高斯filter</option>
                    </select>`;
        }else if(type==='sharpen'){
            html = `<select id="sharpen-method">
                        <option value="roberts">Robert算子</option>
                        <option value="sobel">Sobel算子</option>
                    </select>`;
        }else if(type==='filter'){
            html = `<select id="filter-method">
                        <option value="mean">均值</option>
                        <option value="median">中值</option>
                        <option value="max">最大值</option>
                        <option value="min">最小值</option>
                    </select>`;
        }else if(type==='dl'){
            html = `<select id="dl-method">
                        <option value="beauty">人像美颜</option>
                        <option value="style">风格迁移</option>
                    </select>`;
        }else if(type==='lowpass'){
            html = `<select id="lowpass-method">
                        <option value="ilpf">理想低通滤波器（ILPF）</option>
                        <option value="blpf">巴特沃斯低通滤波器（BLPF）</option>
                        <option value="glpf">高斯低通滤波器（GLPF）</option>
                    </select>
                    <label>截止频率D0: <input type="number" id="lowpass-d0" value="30" min="1" max="256" style="width:60px"></label>
                    <label id="lowpass-n-label" style="display:none;">阶数n: <input type="number" id="lowpass-n" value="2" min="1" max="10" style="width:40px"></label>`;
        }else if(type==='highpass'){
            html = `<select id="highpass-method">
                        <option value="ihpf">理想高通滤波器（IHPF）</option>
                        <option value="bhpf">巴特沃斯高通滤波器（BHPF）</option>
                        <option value="ghpf">高斯高通滤波器（GHPF）</option>
                    </select>
                    <label>截止频率D0: <input type="number" id="highpass-d0" value="30" min="1" max="256" style="width:60px"></label>
                    <label id="highpass-n-label" style="display:none;">阶数n: <input type="number" id="highpass-n" value="2" min="1" max="10" style="width:40px"></label>`;
        }
        processingParams.innerHTML = html;
        if(type==='lowpass'){
            setTimeout(()=>{
                document.getElementById('lowpass-method').addEventListener('change', function(){
                    document.getElementById('lowpass-n-label').style.display = this.value==='blpf' ? '' : 'none';
                });
            },0);
        }
        if(type==='highpass'){
            setTimeout(()=>{
                document.getElementById('highpass-method').addEventListener('change', function(){
                    document.getElementById('highpass-n-label').style.display = this.value==='bhpf' ? '' : 'none';
                });
            },0);
        }
    }
    if(processingType){
        processingType.addEventListener('change', updateProcessingParams);
        updateProcessingParams();
    }

    // 图像处理 - 应用按钮
    document.getElementById('apply-processing').onclick = function(){
        if(!uploadedFilenames.length) return alert('请先上传图片');
        let formData = new FormData();
        formData.append('action', 'processing');
        formData.append('sub_action', processingType.value);
        formData.append('filenames', uploadedFilenames.join(','));
        if(processingType.value === 'gray' && processingParams.firstChild){
            formData.append('param', processingParams.firstChild.value);
        }
        if(processingType.value === 'binary'){
            let threshold = document.getElementById('binary-threshold').value;
            formData.append('threshold', threshold);
        }
        if(processingType.value === 'smooth' && processingParams.firstChild){
            formData.append('param', processingParams.firstChild.value);
        }
        if(processingType.value === 'sharpen' && processingParams.firstChild){
            formData.append('param', processingParams.firstChild.value);
        }
        if(processingType.value === 'filter' && processingParams.firstChild){
            formData.append('param', processingParams.firstChild.value);
        }
        if(processingType.value === 'dl' && processingParams.firstChild){
            formData.append('param', processingParams.firstChild.value);
        }
        if(processingType.value === 'lowpass'){
            formData.append('param', document.getElementById('lowpass-method').value);
            formData.append('d0', document.getElementById('lowpass-d0').value);
            if(document.getElementById('lowpass-method').value==='blpf'){
                formData.append('n', document.getElementById('lowpass-n').value);
            }
        }
        if(processingType.value === 'highpass'){
            formData.append('param', document.getElementById('highpass-method').value);
            formData.append('d0', document.getElementById('highpass-d0').value);
            if(document.getElementById('highpass-method').value==='bhpf'){
                formData.append('n', document.getElementById('highpass-n').value);
            }
        }
        showProgress();
        fetch('/process', {method:'POST', body:formData})
            .then(r=>r.json())
            .then(res=>{
                hideProgress();
                if(res.processed_filename){
                    let url = '/processed/' + res.processed_filename + '?t=' + Date.now();
                    document.getElementById('processed-img-processing').src = url;
                    document.getElementById('download-btn').disabled = false;
                }
            })
            .catch(()=>{
                hideProgress();
                alert('请求出错，请重试');
            });
    };

    // 图像分割类型联动
    const segmentationCategory = document.getElementById('segmentation-category');
    const segmentationType = document.getElementById('segmentation-type');
    function updateSegmentationTypes() {
        let cat = segmentationCategory.value;
        let html = '';
        if (cat === 'traditional') {
            html += `<option value="threshold">阈值检测（传统）</option>
                     <option value="edge">边缘检测（传统）</option>`;
        } else if (cat === 'dl') {
            html += `<option value="maskrcnn">Mask R-CNN实例分割</option>
                     <option value="unet">U-Net语义分割</option>`;
        }
        segmentationType.innerHTML = html;
    }
    if (segmentationCategory && segmentationType) {
        segmentationCategory.addEventListener('change', updateSegmentationTypes);
        updateSegmentationTypes();
    }

    // 图像分割区
    document.getElementById('apply-segmentation').onclick = function(){
        if(!uploadedFilenames.length) return alert('请先上传图片');
        let category = document.getElementById('segmentation-category').value;
        let type = document.getElementById('segmentation-type').value;
        let formData = new FormData();
        formData.append('action', 'segmentation');
        formData.append('sub_action', type);
        formData.append('filenames', uploadedFilenames.join(','));
        if(type === 'threshold') formData.append('threshold', 128);
        showProgress();
        fetch('/process', {method:'POST', body:formData})
            .then(r=>r.json())
            .then(res=>{
                hideProgress();
                if(res.processed_filename){
                    let url = '/processed/' + res.processed_filename + '?t=' + Date.now();
                    document.getElementById('processed-img-segmentation').src = url;
                }
            })
            .catch(()=>{
                hideProgress();
                alert('请求出错，请重试');
            });
    };

    // restoration-params参数区域联动
    const restorationType = document.getElementById('restoration-type');
    const restorationParams = document.getElementById('restoration-params');
    function updateRestorationParams() {
        let type = restorationType.value;
        let html = '';
        if(type === 'deblur_blind') {
            html = `<label>PSF长度: <input type="number" id="psf-length" value="15" min="1" max="50" style="width:50px"></label>
                    <label>PSF角度: <input type="number" id="psf-angle" value="0" min="0" max="180" style="width:50px"></label>
                    <label>迭代数: <input type="number" id="iterations" value="30" min="1" max="100" style="width:50px"></label>`;
        }
        restorationParams.innerHTML = html;
    }
    if(restorationType){
        restorationType.addEventListener('change', updateRestorationParams);
        updateRestorationParams();
    }

    // 多场景图像复原
    document.getElementById('apply-restoration').onclick = function(){
        if(!uploadedFilenames.length) return alert('请先上传图片');
        let type = document.getElementById('restoration-type').value;
        let formData = new FormData();
        formData.append('action', 'restoration');
        if(type === 'dehaze'){
            formData.append('sub_action', 'dehaze');
        }else if(type === 'deblur_wiener'){
            formData.append('sub_action', 'deblur');
            formData.append('method', 'wiener');
        }else if(type === 'deblur_blind'){
            formData.append('sub_action', 'deblur');
            formData.append('method', 'blind');
            formData.append('psf_length', document.getElementById('psf-length').value);
            formData.append('psf_angle', document.getElementById('psf-angle').value);
            formData.append('iterations', document.getElementById('iterations').value);
        }
        formData.append('filenames', uploadedFilenames.join(','));
        showProgress();
        fetch('/process', {method:'POST', body:formData})
            .then(r=>r.json())
            .then(res=>{
                hideProgress();
                if(res.processed_filename){
                    let url = '/processed/' + res.processed_filename + '?t=' + Date.now();
                    document.getElementById('processed-img-restoration').src = url;
                }else if(res.error){
                    alert(res.error);
                }
            })
            .catch(()=>{
                hideProgress();
                alert('请求出错，请重试');
            });
    };

    // 图像拼接与融合区图片上传
    let stitchingFilenames = [];
    const stitchingUpload = document.getElementById('stitching-upload');
    if(stitchingUpload){
        stitchingUpload.addEventListener('change', function(){
            let files = this.files;
            if(files.length < 2){
                alert("请选择两张图片");
                return;
            }
            let formData = new FormData();
            for(let i=0; i<files.length; i++){
                formData.append('file', files[i]);
            }
            showProgress();
            fetch('/upload', {method:'POST', body:formData})
                .then(r=>r.json())
                .then(res=>{
                    hideProgress();
                    if(res.filenames && res.filenames.length>=2){
                        stitchingFilenames = res.filenames.slice(0,2);
                        document.getElementById('original-img-stitching-1').src = '/uploads/' + stitchingFilenames[0];
                        document.getElementById('original-img-stitching-2').src = '/uploads/' + stitchingFilenames[1];
                    }
                })
                .catch(()=>{
                    hideProgress();
                    alert('上传失败，请重试');
                });
        });
    }

    // 图像拼接与融合应用按钮
    document.getElementById('apply-stitching').onclick = function(){
        if(!stitchingFilenames.length || stitchingFilenames.length<2) return alert('请先上传两张图片');
        let type = document.getElementById('stitching-type').value;
        let formData = new FormData();
        formData.append('action', 'stitching');
        formData.append('sub_action', type);
        formData.append('filenames', stitchingFilenames.join(','));
        showProgress();
        fetch('/process', {method:'POST', body:formData})
            .then(r=>r.json())
            .then(res=>{
                hideProgress();
                if(res.processed_filename){
                    let url = '/processed/' + res.processed_filename + '?t=' + Date.now();
                    document.getElementById('processed-img-stitching').src = url;
                }
                else if(res.error){
                    alert(res.error);
                }
            })
            .catch(()=>{
                hideProgress();
                alert('请求出错，请重试');
            });
    };

    // 下载按钮
    document.getElementById('download-btn').onclick = function(){
        let img = document.getElementById('processed-img-processing');
        if(!img.src) return;
        let processedFilename = img.src.split('/').pop().split('?')[0];
        window.open('/download/' + processedFilename, '_blank');
    };
});