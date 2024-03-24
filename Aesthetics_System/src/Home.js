import React from 'react';
import Previews from './components/Previews';
import {Row,Button} from "shards-react";
import PageTitle from "./components/PageTitle";

import {minpx,leftminpx} from './util/utils';


class Home extends React.Component{
	constructor(props) {
		super(props);
	}

	render(){
		return(
			<div style={{
				background:"#FFFFFF",
				paddingLeft:"20px",
				paddingRight:"20px",
				paddingBottom:"40px",
				// width:"1000px",
				position:"absolute",
				left:"50%",
				// marginLeft:"-500px",
				width:minpx(document.body.clientWidth,1000),
				marginLeft:leftminpx(document.body.clientWidth,1000),
			}}>
				<Row noGutters className="page-header py-4">
				<PageTitle title="美学评分系统" subtitle="传音&BUPT" className="text-sm-left mb-3" />
				</Row>
				<Previews/>
			</div>
		);
	}
}
 
export default Home;