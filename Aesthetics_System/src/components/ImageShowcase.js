import React, { Component } from 'react';
import './imageShowcase.css';
// import imgURL from '/Users/lijialong/Desktop/color_system/hue.png';

class ImageShowcase extends Component {
  constructor(props) {
    super(props);
    this.state = {
      colorReady : this.props.colorReady,
      colorFinish : false,
      bgC: "",
      K: 6,
      isMounted: false,
      clusterColors:[],
      showSave:false,
      score:'',
      image: '',
      showElem:'none',
      colorData : this.props.colorData,
      buttonShow : 'block',
      // canvasShowcase : this.props.colorData,
    };
    this.isLoaded = false;
    this.handleOpenClick = this.handleOpenClick.bind(this);
    this.handleClearClick = this.handleClearClick.bind(this);
    this.handleKInput = this.handleKInput.bind(this);
    this.handleKBlur = this.handleKBlur.bind(this);
    this.handleCensusClick = this.handleCensusClick.bind(this);
    this.readFile = this.readFile.bind(this);
    this.drawColor = this.drawColor.bind(this);
    this.handleSaveClick = this.handleSaveClick.bind(this);
    this.handleCanvasClick = this.handleCanvasClick.bind(this);

    // this.colorWheel();
  }

  colorWheel(){
    this.readFile();
    this.handleCensusClick();
  }

  componentDidMount(){
    let pixelRatio = window.devicePixelRatio || 1;
    this.pixelRatio = pixelRatio;
    let canvas = this.canvasShowcase;
    canvas.width =  pixelRatio * parseInt(getComputedStyle(canvas).width);
    canvas.height = pixelRatio * parseInt(getComputedStyle(canvas).height);
    this.oriWidth = canvas.width;
    this.oriHeight = canvas.height;
    this.ctx = canvas.getContext('2d');
    console.log('ctx',this.ctx)
    setTimeout(()=>{
      this.setState({isMounted:true});
    }, 500);
  }

  componentDidUpdate(prevProps, prevState){
    if(prevState.K === this.state.K){
      this.drawPalette();
    }
  }

  handleOpenClick() {
    // this.imgInput.click();
    this.colorWheel();
    
  }

  handleClearClick(){
    this.isLoaded = false;
    this.resetShowcase();
    this.props.resetApp();
  }

  handleKInput(e){
    this.setState({K: e.target.value});
  }

  handleKBlur(e){
    let val = e.target.value;
    val = val<3?3:(val>20?20:val);
    this.setState({K: val});
  }

  handleCensusClick(){
    if(!this.isLoaded){
      return;
    }
    let canvas = this.canvasShowcase;
    console.log('canvas',canvas)
    this.props.setScoreLayer(true);
    this.predictImage(this.state.image);
    setTimeout(()=>{
      this.props.censusColors(this.ctx, this.state.K, canvas.width, canvas.height, this.isHorizontal, this.state.image);
      this.setState({
        buttonShow:'none'
      })
    },600);
  }

  predictImage(image) {
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(image)
    })
      .then(resp => {
        if (resp.ok)
          resp.json().then(data => {
            this.setState({
              score : data.result,
              showElem:'block',
            })
          });
      })

      .catch(err => {
        console.log("An error occured", err.message);
        window.alert("Oops! Something went wrong.");
      });
  }

  handleSaveClick(){
    let canvas = this.canvasShowcase;
    let img = new Image();
    img.src = canvas.toDataURL("image/png");
    let w = window.open('about:blank', 'image from canvas');
    w.document.body.appendChild(img);
  }

  handleCanvasClick(){
    this.setState({
      showSave: !this.state.showSave
    });
    if(this.timer){
        clearTimeout(this.timer);
    }
    this.timer = setTimeout(()=>{
      this.setState({
        showSave: false
      });
    },2300);
  }

  drawColor(main_color, cluster_colors){
      this.setState({
         bgC: main_color,
         clusterColors: cluster_colors
      });
  }

  readFile(){
    this.drawToCanvas(this.state.colorData);
      // let file = this.imgInput.files[0];
      // if(!file){
      //   return false;
      // }
      // if (!/image\/\w+/.test(file.type)) {
      //   console.log("image needed!");
      //   return false;
      // }
      // let reader = new FileReader();
      
      // reader.readAsDataURL(file); //转化成base64数据类型
      // // todo 这里可以获取图片
      // let that = this;
      // reader.onload = function() {
      //   that.drawToCanvas(that.state.colorData);
      //   console.log('image',this.result);
      //   that.setState({
      //     image : this.result,
      //   })
      // }
  }

  resetShowcase(){
     let pixelRatio = this.pixelRatio;
     let canvas = this.canvasShowcase;
     canvas.width = this.oriWidth;
     canvas.style.width = this.oriWidth/pixelRatio + "px";
     canvas.height = this.oriHeight;
     canvas.style.height = this.oriHeight/pixelRatio + "px";
     this.setState({
      bgC: "",
      score:'',
      showElem:'none',
      clusterColors: []
     });
  }

  drawToCanvas(imgData){
    // this.handleClearClick();
    let pixelRatio = this.pixelRatio;
    this.isLoaded = true;
    let canvas = this.canvasShowcase;
    console.log('canvas',canvas)
    let c_w = canvas.width;
    let c_h = canvas.height;
    let ctx = this.ctx;
    let img = new Image();
    img.src = imgData;
    img.onload = () => {
      ctx.clearRect(0, 0, c_w, c_h);
      let _w = 0;
      let _h = 0;
      if(img.width<img.height){
        _w = 100*pixelRatio;
        this.isHorizontal = false;
      }else{
        _h = 100*pixelRatio;
        this.isHorizontal = true;
      }
      let img_w = img.width > (c_w-_w)/pixelRatio ? (c_w-_w)/pixelRatio : img.width;
      let img_h = img.height > (c_h-_h)/pixelRatio ? (c_h-_h)/pixelRatio : img.height;
      let scale = (img_w / img.width < img_h / img.height) ? (img_w / img.width) : (img_h / img.height);
      img_w = img.width * scale;
      img_h = img.height * scale;
      console.log(img_w,img_h)
      canvas.style.width = img_w + _w/pixelRatio + "px";
      canvas.style.height = img_h + _h/pixelRatio + "px";
      canvas.width = (img_w*pixelRatio + _w);
      canvas.height = (img_h*pixelRatio + _h);
      ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, img_w*pixelRatio, img_h*pixelRatio);
    };
  }

  drawPalette(){
    let pixelRatio = this.pixelRatio;
    let canvas = this.canvasShowcase;
    let c_w = canvas.width;
    let c_h = canvas.height;
    let ctx = this.ctx;
    let K = this.state.K;
    let len = this.isHorizontal ? c_w : c_h;
    let interval = len*(K<10 ? 0.02 : 0.01);
    // interval *= pixelRatio;
    let p = (len-(K-1)*interval) / K;
    let colors = this.state.clusterColors;
    if(colors.length===0){
      return;
    }
    if(this.isHorizontal){
      ctx.clearRect(0, c_h - 90*pixelRatio, c_w, 90*pixelRatio);
    }else{
      ctx.clearRect(c_w - 90*pixelRatio, 0, 90*pixelRatio, c_h);
    }
    for(let i=0;i<K;i++){
      ctx.fillStyle = colors[i];
      if(this.isHorizontal){
        ctx.fillRect((p + interval)*i ,c_h - 90*pixelRatio,p,90*pixelRatio);
      }else{
        ctx.fillRect(c_w - 90*pixelRatio, (p + interval)*i, 90*pixelRatio, p);
      }
    }
  }

  render() {
    let showcaseClass = this.state.isMounted ? 'mounted' : '';
    showcaseClass += " image-showcase";
    let showcaseWrapClass = this.state.bgC ? 'censused' : '';
    showcaseWrapClass += " showcase-wrap";
    let saveWrapClass = this.state.showSave ? 'show' : '';
    saveWrapClass += " save-wrap";
    let score = this.state.score ? '色彩得分 : ' : '';
    score += this.state.score;


    console.log("img",this.props.colorReady)

    if (this.props.colorReady && !this.state.colorFinish)
    {
      this.colorWheel();
      this.setState({colorFinish: true})
    }

    return (
      <div className={showcaseClass} style={{display:this.state.buttonShow}}>
        {/* <div className="head"><b>Color evaluation</b></div> */}
        <div className={showcaseWrapClass} >
          <canvas  className="showcase" ref={(canvas) => { this.canvasShowcase = canvas; }}></canvas>
        </div>
        {/*<div className="allButton"  >*/}
        
        {/*  /!* <input className="img-input" type="file" multiple="multiple" accept="image/*" onChange={this.readFile} ref={(input) => { this.imgInput = input; }}/> *!/*/}
        {/*  <input type="button" value="色彩组成" className="button" onClick={this.handleOpenClick}/>*/}
        {/*  /!* <input type="button" value="clear" className="button" onClick={this.handleClearClick} /> */}
        {/*  <input type="button" value="submit" className="button" onClick={this.handleCensusClick} />  *!/*/}
        {/*</div>*/}
        {/* <div className="imgShow">
          <img style={{display:this.state.showElem}} src={imgURL} className="ringImg"></img>  */}
          {/* <img src={this.state.image}  ></img>
        </div>
        
        {/* <div className="score"><b>{score}</b></div> */}
        {/* <button onClick={this.handleOpenClick} title="open image" >open image</button>
        <button onClick={this.handleClearClick} title="clear image" >clean image</button>
        <button onClick={this.handleCensusClick} title="census color" >evaluate image</button> */}
        {/* <input type="number"  className="input-K"  value={this.state.K} onBlur={this.handleKBlur} onChange={this.handleKInput} title="K means" /> */}
        {/* <div className={saveWrapClass}>
          <span onClick={this.handleSaveClick} className="save">SAVE</span>
        </div> */}
      </div>
    );
  }
}

export default ImageShowcase;
