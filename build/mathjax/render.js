const fs=require('fs');
const {mathjax}=require('mathjax-full/js/mathjax.js');
const {TeX}=require('mathjax-full/js/input/tex.js');require('mathjax-full/js/input/tex/AllPackages.js');
const {SVG}=require('mathjax-full/js/output/svg.js');
const {liteAdaptor}=require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler}=require('mathjax-full/js/handlers/html.js');
const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);
const doc=mathjax.document('',{InputJax:new TeX({packages:['base','ams','newcommand','boldsymbol','color','cancel']}),OutputJax:new SVG({fontCache:'none'})});
const items=JSON.parse(fs.readFileSync(process.argv[2]));const out={};let bad=0;
for(const it of items){
  try{const node=doc.convert(it.tex,{display:it.display,em:16,ex:8,containerWidth:80*16});
    let svg=adaptor.innerHTML(node);
    if(svg.includes('data-mjx-error')||svg.includes('merror')){bad++;console.error('ERR',it.tex);}
    out[it.id]=svg;}catch(e){bad++;console.error('EXC',it.tex,e.message);out[it.id]='';}
}
fs.writeFileSync(process.argv[3],JSON.stringify(out));console.log('rendered',items.length,'bad',bad);
