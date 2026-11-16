import {useState} from 'react';
import type {Box} from './types';
export const colors=['#2dd4bf','#fbbf24','#60a5fa','#fb7185','#c084fc','#a3e635','#fb923c','#38bdf8','#e879f9','#f87171'];
export function Viewer({url,boxes=[],classes=[],threshold,showBoxes,showLabels,showScores,filter,zoom,offset,onOffset,gt=[],showGT,onSelect}:{url:string;boxes?:Box[];classes?:string[];threshold:number;showBoxes:boolean;showLabels:boolean;showScores:boolean;filter:string;zoom:number;offset:{x:number;y:number};onOffset:(p:{x:number;y:number})=>void;gt?:Box[];showGT:boolean;onSelect?:(b:Box)=>void}){
 const [size,setSize]=useState({w:1,h:1});const [drag,setDrag]=useState<{x:number;y:number}|null>(null);
 return <div className="viewport" onPointerDown={e=>{if(zoom>1){e.currentTarget.setPointerCapture(e.pointerId);setDrag({x:e.clientX-offset.x,y:e.clientY-offset.y})}}} onPointerMove={e=>drag&&onOffset({x:e.clientX-drag.x,y:e.clientY-drag.y})} onPointerUp={()=>setDrag(null)} onPointerCancel={()=>setDrag(null)}>
  <div className="image-stage" style={{transform:`translate(${offset.x}px,${offset.y}px) scale(${zoom})`}}>
   <img src={url} alt="Ảnh đang phân tích" draggable={false} onLoad={e=>setSize({w:e.currentTarget.naturalWidth,h:e.currentTarget.naturalHeight})}/>
   <svg viewBox={`0 0 ${size.w} ${size.h}`} className="boxes" aria-label="Bounding boxes">
    {showGT&&gt.filter(b=>filter==='all'||b.label===filter).map((b,i)=><rect key={'gt'+i} x={b.bbox[0]} y={b.bbox[1]} width={b.bbox[2]-b.bbox[0]} height={b.bbox[3]-b.bbox[1]} fill="none" stroke="#fff" strokeWidth={2/zoom} strokeDasharray="7 5" vectorEffect="non-scaling-stroke"/>) }
    {showBoxes&&boxes.filter(b=>b.score>=threshold&&(filter==='all'||b.label===filter)).map((b,i)=>{const c=colors[Math.max(0,classes.indexOf(b.label))%colors.length];const font=Math.max(12,size.w/65);const text=`${showLabels?b.label:''}${showScores?' '+Math.round(b.score*100)+'%':''}`;return <g key={i} className="box-group" onPointerDown={e=>e.stopPropagation()} onClick={()=>onSelect?.(b)}><rect x={b.bbox[0]} y={b.bbox[1]} width={Math.max(0,b.bbox[2]-b.bbox[0])} height={Math.max(0,b.bbox[3]-b.bbox[1])} stroke={c} strokeWidth="2" fill="transparent" vectorEffect="non-scaling-stroke"/>{text.trim()&&<><rect x={b.bbox[0]} y={Math.max(0,b.bbox[1]-font*1.6)} width={text.length*font*.61+10} height={font*1.6} fill={c} rx="3"/><text x={b.bbox[0]+5} y={Math.max(font,b.bbox[1]-font*.4)} fontSize={font} fill="#071217" fontWeight="700">{text}</text></>}</g>})}
   </svg>
  </div>
 </div>
}
