import React from "react";
import PropTypes from "prop-types";
import { Container, Row, Col, Button } from "shards-react";
import { Card, CardBody } from "shards-react";


import PageTitle from "./components/PageTitle";
import DetailCards from "./components/DetailCards";


class BlogOverview extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    const {
      header1, image1, footer1,
      header2, image2, footer2,
      imageD} = this.props;

    return (
      <div style={{width:"800px",position:"absolute",left:"50%",marginLeft:"-400px"}}>
        <Row noGutters className="page-header py-4">
          <PageTitle title="图片报告" subtitle="传音&BUPT" className="text-sm-left mb-3" />
        </Row>
        <Row>
          <div style={{position:"relative", left:"15px",bottom:"10px"}}>
            <Button pill>
              上传图片
            </Button>
            <Button pill style={{position:"relative", left:"15px"}}>
              生成报告
            </Button>
          </div>
        </Row>

        <Row>
          <Col md="8" className="mb-4">
            <DetailCards
              noFooter={false}
              bodyIsImg={true}
              header={header1}
              dataBody={image1}
              footer={footer1}/>
          </Col>
          <Col md="4" className="mb-4">
            <DetailCards
              bodyIsImg={true}
              header={header2}
              dataBody={imageD}
              imgHeight={""}/>
          </Col>
        </Row>

        <Row>
          <Col md="12" className="mb-4">
            {/*<Card>*/}
            {/*  <CardBody>*/}
                <Row>
                  <Col md="2">
                    <DetailCards
                      noFooter={false}
                      header={"色彩评分"}
                      dataBody={"6.54"}
                      footer={"色彩搭配恰当"}/>
                  </Col>
                  <Col md="5">
                    <DetailCards
                      bodyIsImg={true}
                      header={"色环"}
                      dataBody={imageD}/>
                  </Col>
                  <Col md="5">
                    <DetailCards
                      bodyIsImg={true}
                      header={"配色"}
                      dataBody={imageD}/>
                  </Col>
                </Row>
              {/*</CardBody>*/}
            {/*</Card>*/}
          </Col>
        </Row>

        <Row>
          <Col md="2" className="mb-4">
            <DetailCards
              noHeader={false}
              bodyIsImg={false}
              noFooter={true}
              header={"曝光评分"}
              dataBody={"6.54"}
              footer={""}/>
          </Col>
          <Col md="5" className="mb-4">
            <DetailCards
              noHeader={false}
              bodyIsImg={true}
              noFooter={true}
              header={"曝光热力图"}
              dataBody={imageD}/>
          </Col>
          <Col md="5" className="mb-4">
            <DetailCards
              noHeader={false}
              bodyIsImg={false}
              noFooter={true}
              header={"曝光舒适度评分"}
              dataBody={"？分"}/>
          </Col>
        </Row>
        <Row>
          <Col md="2" className="mb-4">
            <DetailCards
              noHeader={false}
              bodyIsImg={false}
              noFooter={true}
              header={"构图评分"}
              dataBody={"6.54"}
              footer={""}/>
          </Col>
          <Col md="10" className="mb-4">
            <DetailCards
              noHeader={false}
              bodyIsImg={false}
              noFooter={true}
              header={"构图分类"}
              dataBody={"井字形构图"}/>
          </Col>
        </Row>
        <Row>
          <Col md="2" className="mb-4">
            <DetailCards
              noHeader={false}
              bodyIsImg={false}
              noFooter={true}
              header={"对焦评分"}
              dataBody={"6.54"}
              footer={""}/>
          </Col>
          <Col md="10" className="mb-4">
            <DetailCards
              noHeader={false}
              bodyIsImg={false}
              noFooter={true}
              header={"对焦类型判断"}
              dataBody={"全焦"}/>
          </Col>
        </Row>
        <Row>
          <Col md="2" className="mb-4">
            <DetailCards
              noHeader={false}
              bodyIsImg={false}
              noFooter={true}
              header={"噪声评分"}
              dataBody={"6.54"}
              footer={""}/>
          </Col>
          <Col md="10" className="mb-4">
            <DetailCards
              noHeader={false}
              bodyIsImg={false}
              noFooter={true}
              header={"balabala"}
              dataBody={"balabala"}/>
          </Col>
        </Row>
      </div>
    );
  }
}

BlogOverview.propTypes = {
};

BlogOverview.defaultProps = {
  header1: "图片预览",
  image1: require("./images/default/main.png"),
  footer1: "综合分数: 0.00",
  header2: "蛛网图",
  image2: require("./images/default/non.jpg"),
  footer2: "",
  imageD: require("./images/default/non.jpg")
};

export default BlogOverview;
