import React from "react";
import PropTypes from "prop-types";
import classNames from "classnames";
import {Card, CardBody, CardFooter, CardHeader, Col, Row} from "shards-react";
import EchartsRadar from "./EchartsRadar";



class RadarCard extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    const header = this.props.header;
    const footer = this.props.footer;
    const noFooter = this.props.noFooter;

    const mainScore = this.props.mainScore;
    const colorScore = this.props.colorScore;
    const exposureScore = this.props.exposureScore;
    const noiseScore = this.props.noiseScore;
    const compositionScore = this.props.compositionScore;
    const iqaScore = this.props.iqaScore;


    return (
      <Card small className="h-100">
        <CardHeader className="bg-dark">
          <h6 className="m-0">
            <font color="white">{header}</font>
          </h6>
        </CardHeader>

        <CardBody align="center">
          <EchartsRadar
            mainScore = {mainScore}
            colorScore = {colorScore}
            exposureScore = {exposureScore}
            noiseScore = {noiseScore}
            compositionScore={compositionScore}
            iqaScore={iqaScore}
          />
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

RadarCard.propTypes = {
  noFooter: PropTypes.bool,
};

RadarCard.defaultProps = {
  noFooter: true,
};

export default RadarCard;
