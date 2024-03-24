import React from "react";
import PropTypes from "prop-types";
import {Card, CardBody, Row, Col, Button, CardHeader} from "shards-react";
import { Spin } from 'antd';

import PageTitle from "./components/PageTitle";
import DetailCards from "./components/DetailCards";
import EchartsRadar from './components/EchartsRadar';
import RadarCard from "./components/RadarCard";
import ColorCensus from "./components/ColorCensus";
import Pdf from "./components/Pdf";
import {minpx,leftminpx} from './util/utils';

class Report extends React.Component {
  constructor(props) {
    super(props);

    // Get base64 data of img, from Function Previews
    const data = this.props.location.state.data;
    console.log(this.props.location);
    console.log(data)

    this.state = {
      defaultImage: props.defaultImage,

      mainHeader: props.mainHeader,
      mainImage: data,
      mainScore: props.mainScore,
      mainDescription: props.mainDescription,

      radarIsReady: props.radarIsReady,
      radarHeader: props.radarHeader,
      radarImage: props.radarImage,
      radarFooter: props.radarFooter,

      colorScore: props.colorScore,
      colorReady: props.colorReady,
      colorDescription: props.colorDescription,

      exposureScore:props.exposureScore,
      exposureHeatMap:props.exposureHeatMap,
      exposureDescription: props.exposureDescription,

      noiseScore:props.noiseScore,
      noiseDescription: props.noiseDescription,

      compositionScore:props.compositionScore,
      compositionTypePre:props.compositionTypePre,
      compositionDescription:props.compositionDescription,

      iqaScore:props.iqaScore,
      iqaDescription: props.iqaDescription,

      modelLoaded: props.modelLoaded,
    };

    this.predictImage(this.state.mainImage)
  }

  render() {
    let {
      mainHeader, mainImage, mainScore, mainDescription,
      radarIsReady, radarHeader, radarImage, radarFooter,
      colorScore, colorDescription,
      exposureScore, exposureHeatMap, exposureDescription,
      noiseScore, noiseDescription,
      compositionScore, compositionTypePre, compositionDescription,
      iqaScore, iqaDescription,
      defaultImage,
      modelLoaded,
    } = this.state;

    console.log("repo",this.state.colorReady)

    return (
      <div id = "report"
        style={{
          // background:"#FFFFFF",
          paddingLeft:"20px",
          paddingRight:"20px",
          paddingBottom:"40px",
          // width:"1000px",
          position:"absolute",
          left:"50%",
          // marginLeft:"-500px",
          width:minpx(document.body.clientWidth,1000),
          marginLeft:leftminpx(document.body.clientWidth,1000),
        }}
      >
        <div >
         <Spin spinning={!modelLoaded} tip="评分中......" size="large">
          <Row noGutters className="page-header py-4">
            <PageTitle title="图片报告" subtitle="传音&BUPT" className="text-sm-left mb-3" />
          </Row>
          {/* <Pdf/> */}
          {/*测试用*/}
          {/*<Row>*/}
          {/*  <Col md="8" className="mb-4">*/}
          {/*    <input onChange={() => this.onChange()}*/}
          {/*           accept="image/gif,image/jpeg,image/jpg,image/png,image/svg"*/}
          {/*           id="input"*/}
          {/*           type="file" />*/}
          {/*  </Col>*/}
          {/*</Row>*/}

          {/*<Row>*/}
          {/*  <div style={{position:"relative",bottom:"10px"}}>*/}
          {/*    /!*<Button pill*!/*/}
          {/*    /!*        onClick={() => this.upload()}>*!/*/}
          {/*    /!*  选择图片*!/*/}
          {/*    /!*</Button>*!/*/}
          {/*    <Button pill*/}
          {/*            style={{position:"relative", left:"15px"}}*/}
          {/*            onClick={() => this.predictImage(mainImage)}>*/}
          {/*      上传图片*/}
          {/*    </Button>*/}
          {/*    <Button pill*/}

          {/*            style={{position:"relative", left:"30px"}}>*/}
          {/*      生成报告*/}
          {/*    </Button>*/}
          {/*  </div>*/}
          {/*</Row>*/}
		  
          <Row>
            <Col md="8" className="mb-4">
              <DetailCards
                noFooter={false}
                bodyIsImg={true}
                header={mainHeader}
                dataBody={mainImage}
                footer={"综合分数: " + mainScore}/>
            </Col>
            <Col md="4" className="mb-4">
              {!radarIsReady &&
              <DetailCards
                noFooter={true}
                header={radarHeader}
                dataBody={""}/>
              }
              {radarIsReady &&
              <RadarCard
                header={radarHeader}
                mainScore={mainScore}
                colorScore={colorScore}
                noiseScore={noiseScore}
                exposureScore={exposureScore}
                compositionScore={compositionScore}
                iqaScore={iqaScore}
              />
              }
            </Col>
          </Row>

          <Row className="mb-4">
            <Col>
              <Card small>
                <CardHeader className="bg-dark">
                  <h6 className="m-0">
                    <font color="white">整体美学评分</font>
                  </h6>
                </CardHeader>
                <CardBody className="p-0 pb-3">
                  <div style={{marginLeft:"20px",marginRight:"20px",
                    marginTop:"10px",marginBottom:"10px"}}>
                    <p>
                      <b>Score: </b>
                      <font color="#1e90ff">{mainScore}</font>
                    </p>
                    <p>
                      <b>
                        {mainDescription}
                      </b>
                    </p>
                  </div>
                  {/* <div style={{marginLeft:"20px",marginRight:"20px"}}>
                  </div> */}
                </CardBody>
              </Card>
            </Col>
          </Row>

          <Row className="mb-4">
            <Col>
              <Card small>
                <CardHeader className="bg-dark">
                  <h6 className="m-0">
                    <font color="white">色彩评分</font>
                  </h6>
                </CardHeader>
                <CardBody className="p-0 pb-3">
                  <div style={{marginLeft:"20px",marginRight:"20px",
                    marginTop:"10px",marginBottom:"10px"}}>
                    <p>
                      <b>Score: </b>
                      <font color="#1e90ff">{colorScore}</font>
                    </p>
                    <p>
                      <b>
                        {colorDescription}
                      </b>
                    </p>
                  </div>
                  <div style={{marginLeft:"20px",marginRight:"20px"}}>
                    {/*<table className="table mb-0">*/}
                    {/*  <thead className="bg-light">*/}
                    {/*  <tr>*/}
                    {/*    <th scope="col" className="border-0">*/}
                    {/*      一列*/}
                    {/*    </th>*/}
                    {/*    <th scope="col" className="border-0">*/}
                    {/*      又一列*/}
                    {/*    </th>*/}
                    {/*    <th scope="col" className="border-0">*/}
                    {/*      再一列*/}
                    {/*    </th>*/}
                    {/*    <th scope="col" className="border-0">*/}
                    {/*      还来一列*/}
                    {/*    </th>*/}
                    {/*  </tr>*/}
                    {/*  </thead>*/}
                    {/*  <tbody>*/}
                    {/*  <tr>*/}
                    {/*    <td>色彩</td>*/}
                    {/*    <td>3.33</td>*/}
                    {/*    <td>???</td>*/}
                    {/*    <td>###</td>*/}
                    {/*  </tr>*/}
                    {/*  <tr>*/}
                    {/*    <td>色彩</td>*/}
                    {/*    <td>3.33</td>*/}
                    {/*    <td>???</td>*/}
                    {/*    <td>###</td>*/}
                    {/*  </tr>*/}
                    {/*  <tr>*/}
                    {/*    <td>色彩</td>*/}
                    {/*    <td>3.33</td>*/}
                    {/*    <td>???</td>*/}
                    {/*    <td>###</td>*/}
                    {/*  </tr>*/}
                    {/*  </tbody>*/}
                    {/*</table>*/}
                  </div>
                  <div align="center">
                      <ColorCensus colorData={this.state.mainImage}
                                   colorReady={this.state.colorReady}/>
                  </div>
                </CardBody>
              </Card>
            </Col>
          </Row>
            
          <Row className="mb-4">
            <Col>
              <Card small>
                <CardHeader className="bg-dark">
                  <h6 className="m-0">
                    <font color="white">曝光评分</font>
                  </h6>
                </CardHeader>
                <CardBody className="p-0 pb-3">
                  <div style={{marginLeft:"20px",marginRight:"20px",
                    marginTop:"10px",marginBottom:"10px"}}>
                    <p>
                      <b>Score: </b>
                      <font color="#1e90ff">{exposureScore}</font>
                    </p>
                    <p>
                      <b>
                        {exposureDescription}
                      </b>
                    </p>
                    <div align="center">
                      <img height={"150px"} src={exposureHeatMap}/>
                    </div>
                  </div>
                  {/* <div style={{marginLeft:"20px",marginRight:"20px"}}>
                  </div> */}
                </CardBody>
              </Card>
            </Col>
          </Row>

          <Row className="mb-4">
            <Col>
              <Card small>
                <CardHeader className="bg-dark">
                  <h6 className="m-0">
                    <font color="white">噪声评分</font>
                  </h6>
                </CardHeader>
                <CardBody className="p-0 pb-3">
                  <div style={{marginLeft:"20px",marginRight:"20px",
                    marginTop:"10px",marginBottom:"10px"}}>
                    <p>
                      <b>Score: </b>
                      <font color="#1e90ff">{noiseScore}</font>
                    </p>
                    <p>
                      <b>
                        {noiseDescription}
                      </b>
                    </p>
                  </div>
                  {/* <div style={{marginLeft:"20px",marginRight:"20px"}}>
                  </div> */}
                </CardBody>
              </Card>
            </Col>
          </Row>
		  
          <Row className="mb-4">
            <Col>
              <Card small>
                <CardHeader className="bg-dark">
                  <h6 className="m-0">
                    <font color="white">构图评分</font>
                  </h6>
                </CardHeader>
                <CardBody className="p-0 pb-3">
                  <div style={{marginLeft:"20px",marginRight:"20px",
                    marginTop:"10px",marginBottom:"10px"}}>
                    <p>
                      <b>Score: </b>
                      <font color="#1e90ff">{compositionScore}</font>
                    </p>
                    <p>
                      <b>
                        {compositionDescription}
                      </b>
                    </p>
                    <div align="center">
                      <img height={"200px"} src={compositionTypePre}/>
                    </div>
                  </div>
                  {/* <div style={{marginLeft:"20px",marginRight:"20px"}}>
                  </div> */}
                </CardBody>
              </Card>
            </Col>
          </Row>

          <Row className="mb-4">
            <Col>
              <Card small>
                <CardHeader className="bg-dark">
                  <h6 className="m-0">
                    <font color="white">IQA评分</font>
                  </h6>
                </CardHeader>
                <CardBody className="p-0 pb-3">
                  <div style={{marginLeft:"20px",marginRight:"20px",
                    marginTop:"10px",marginBottom:"10px"}}>
                    <p>
                      <b>Score: </b>
                      <font color="#1e90ff">{iqaScore}</font>
                    </p>
                    <p>
                      <b>
                        {iqaDescription}
                      </b>
                    </p>
                  </div>
                  {/* <div style={{marginLeft:"20px",marginRight:"20px"}}>
                  </div> */}
                </CardBody>
              </Card>
            </Col>
          </Row>
		  
          <Row className="mb-4"/>
         </Spin>
        </div>
        <div><Pdf/></div>
      </div>
      
    );
  }

  // 发送Post请求给app.py, 调用模型算分
  predictImage(image)
  {
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
              mainScore: data.mainScore,
              colorScore: data.colorScore,
              exposureScore: data.exposureScore,
              exposureHeatMap: data.exposureHeatMap,
              noiseScore: data.noiseScore,
              compositionScore: data.compositionScore,
              compositionTypePre: data.compositionTypePre,
              iqaScore: data.iqaScore,

              mainDescription: data.mainDescription,
              colorDescription: data.colorDescription,
              exposureDescription: data.exposureDescription,
              noiseDescription: data.noiseDescription,
              compositionDescription: data.compositionDescription,
              iqaDescription: data.iqaDescription,

              radarIsReady: true,
              colorReady: true,
              modelLoaded: true,
            })
          });
      })
      .catch(err => {
        console.log("An error occured", err.message);
        window.alert("Oops! Something went wrong.");
      });
  }

  // // 以下两个函数：测试用，可以在本页面上传图片，获得base64码
  // upload() {
  //   const input = document.getElementById('input')
  //   input.click()
  // }
  //
  // onChange() {
  //   var reader = new FileReader();
  //   //拿到上传的图片
  //   var file = document.getElementById('input').files[0];
  //
  //   if (file) {
  //     let that = this;
  //     reader.readAsDataURL(file);
  //     reader.onload = function (e) {
  //       that.setState({
  //         mainImage: reader.result,
  //       })
  //       console.log(reader.result);
  //     }
  //   }
  // }
}

// 属性值类型(弃用)
Report.propTypes = {
};

// 各属性默认值
Report.defaultProps = {
  // Default resources
  defaultImage: require("./images/default/non.jpg"),

  // Image Preview Card Data
  mainHeader: "图片预览",
  mainImage: require("./images/default/non.jpg"),
  mainScore: "0.00",
  mainDescription: "",

  // Radar Card Data
  radarIsReady: false,
  radarHeader: "蛛网图",
  radarImage: require("./images/default/non.jpg"),
  radarFooter: "",

  // 色彩卡片数据
  colorScore: "0.00",
  colorReady: false,
  colorDescription: "",

  // 曝光卡片数据
  exposureScore: "0.00",
  exposureHeatMap: '',
  // exposureHeatMap: require("./images/Exposure-heat-map.png"),
  exposureDescription: "",

  // 噪声卡片数据
  noiseScore: "0.00",
  noiseDescription: "",

  // 构图卡片数据
  compositionScore: "0.00",
  compositionTypePre: '',
  // compositionTypePre: require("./images/Composition-type-pre.png"),
  compositionDescription: "",

  // IQA卡片数据
  iqaScore: "0.00",
  iqaDescription: "",

  // 模型数据加载完毕flag
  modelLoaded: false,
};

export default Report;

