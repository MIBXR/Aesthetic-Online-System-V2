import React from 'react';
 
class Page1 extends React.Component{
	render(){
		const {data} = this.props.match.params;
		console.log(this.props);

		return(
			<div>
			<div>This is Page1!</div>
			</div>
		);
	}
}
 
export default Page1;