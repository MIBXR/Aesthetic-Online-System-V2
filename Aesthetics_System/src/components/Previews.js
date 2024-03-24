import React, {useEffect, useState} from 'react';
import {useDropzone} from 'react-dropzone';
import {Button, Row ,CardBody} from "shards-react";
import { Route, useHistory } from 'react-router-dom'
import {minpx,leftminpx} from '../util/utils.js';

const thumbsContainer = {
  display: 'flex',
  flexDirection: 'row',
  flexWrap: 'wrap',
  marginTop: 16 ,
  // border: '3px solid red',
};

const ct = {
  border: '2px dashed gray',
  padding: 4,
};

const p = {
  fontSize: 50 ,
  textAlign: 'center',
};

const thumb = {
  display: 'inline-flex',
  borderRadius: 2,
  border: '1px solid #eaeaea',
  marginBottom: 8,
  marginRight: 8,
  // width: 100,
  // height: 100,
  width: 'auto',
  height: 'auto',
  padding: 4,
  boxSizing: 'border-box',
  margin: 'auto',
};

const thumbInner = {
  display: 'flex',
  minWidth: 0,
  overflow: 'hidden'
};

const img = {
  display: 'block',
  width: 600,
  height: '100%'
};


function Previews(props) {
  const history = useHistory();
  // const [data, setData] = useState(null);
  const [files, setFiles] = useState([]);
  const {getRootProps, getInputProps} = useDropzone({
    accept: 'image/*',
    onDrop: acceptedFiles => {
      setFiles(acceptedFiles.map(file => Object.assign(file, {
        preview: URL.createObjectURL(file)
      })));
    }
  });

  // const handleGoTo = () => {
  //   history.push({pathname:'/report', state:{data:data}})
  //   // history.push(`/report/${data}`);
  // }

  // function sleep(time) {
  //   return new Promise((resolve) => setTimeout(resolve, time));
  // }
  
  const upload =()=>{
    const file = files[0]
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = function (result) {
      // setData(this.result);
      history.push({pathname:'/report', state:{data:this.result}})
    }
    // setTimeout(() => {
    //   console.log(dataRef.current)
    // }, 1000);
  }


  const thumbs = files.map(file => (
    <div style={thumb} key={file.name}>
      <div style={thumbInner}>
        <img
          src={file.preview}
          style={img}
        />
      </div>
    </div>
  ));

  useEffect(() => () => {
    // Make sure to revoke the data uris to avoid memory leaks
    files.forEach(file => URL.revokeObjectURL(file.preview));
  }, [files]);

  return (
    <section className="container">
      <div {...getRootProps({className: 'dropzone'})} style={ct}>
        <input {...getInputProps()} />
        <p style={p}>拖放或点击上传图片</p>
      </div>
      <aside style={thumbsContainer}>
        {thumbs}
      </aside>
       <Row>
          <CardBody align="center">
            <Button pill style={{position:"relative", right:"15px"}} onClick={() =>setFiles([])}>清空</Button>
            <Button pill style={{position:"relative", left:"0px"}} onClick={upload}>评测</Button>
            {/* <Button pill style={{position:"relative", left:"15px"}} onClick={handleGoTo}>测评</Button> */}
          </CardBody>
        {/* <Route path="/report" component ={Report} /> */}
      </Row>
    </section>
  );
}

<Previews />
export default Previews;