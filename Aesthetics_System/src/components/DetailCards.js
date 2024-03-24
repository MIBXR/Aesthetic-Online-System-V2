import React from "react";
import PropTypes from "prop-types";
import classNames from "classnames";
import {Card, CardBody, CardFooter, CardHeader, Col, Row} from "shards-react";



class DetailCards extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    const header = this.props.header;
    const dataBody = this.props.dataBody;
    const footer = this.props.footer;

    const noHeader = this.props.noHeader;
    const bodyIsImg = this.props.bodyIsImg;
    const noFooter = this.props.noFooter;

    const imgHeight = this.props.imgHeight;

    return (
      <Card small className="h-100">
        <CardHeader className="bg-dark">
          {!noHeader &&
          <h6 className="m-0">
            <font color="white">{header}</font>
          </h6>}
        </CardHeader>

        <CardBody align="center">
          {bodyIsImg &&
              <img height={imgHeight} src={dataBody}/>
          }
          {!bodyIsImg &&
            <div className="blog-comments__item d-flex p-3">
            <div className="blog-comments__avatar mr-3">
              <h6 className="m-0">{dataBody}</h6>
            </div>
          </div>}
        </CardBody>

        {!noFooter && <CardFooter className="border-top">
          <Row>
            <Col>
              <h7 className="m-0">{footer}</h7>
            </Col>
          </Row>
        </CardFooter>}
      </Card>
    );
  }
}

DetailCards.propTypes = {
  noHeader: PropTypes.bool,
  bodyIsImg: PropTypes.bool,
  noFooter: PropTypes.bool,
  imgHeight: PropTypes.string
};

DetailCards.defaultProps = {
  noHeader: false,
  bodyIsImg: false,
  noFooter: true,
  imgHeight: "200px"
};

export default DetailCards;
