import React from 'react'
import ReactDOM from 'react-dom'
import { BrowserRouter as Router, Link, Route } from 'react-router-dom'
import Home from './Home'
import Page1 from './Page1'
import Report from "./Report";
import BlogOverview from "./BlogOverview";
import Previews from "./components/Previews";
// import {Row} from "shards-react";
// import PageTitle from "./components/PageTitle";

const App = () => (


  <Router>
    {/* <div style={{width:"800px",position:"absolute",left:"50%",marginLeft:"-400px"}}>
        <Row noGutters className="page-header py-4">
          <PageTitle title="美学评分系统" subtitle="传音&BUPT" className="text-sm-left mb-3" />
        </Row>
      <Link to="/report">Report</Link>
      <br/>
      <Link to="/blogoverview">BlogOverview</Link>
      <br/>
      <Link to="/pre">Previews</Link>
        <Route exact path="/" component={Home} />
        <Route path="/report" component={Report} />
        <Route path="/page1" component={Page1} />
        <Route path="/report" component={Report } />
        <Route path="/blogoverview" component={BlogOverview} />
        <Route path="/pre" component={Previews} />
    </div> */}
    <div>
      <Route exact path="/" component={Home} />
      <Route path="/report" component={Report} />
    </div>
  </Router>
)

export default App;