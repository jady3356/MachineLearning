
var fs = require("fs");

var path = require("path");
console.log(__dirname);
//var data = fs.readFileSync(path.join(__dirname, 'poetryfromPi3Testing.txt'));
var data = fs.readFileSync('../../../../root/TF/xieshi/poetryfromPi3Testing0.txt');
//var data = fs.readFile('../../../../root/TF/xieshi/poetryfromPi3Testing.txt', 'utf-8'); 

poetry0 = data.toString();

poetry0 = poetry0.replace(/。/g, "。\n");
//console.log(poetry);
//poetry = poetry.split(/。/);
//const cp=require('child_process')
//console.log(process.cwd());

//cp.exec('python path.resolve(__dirname, '../../../../root/TF/xieshi/getPoetryFromPi3.py')');
//cp.exec('python ../../../../root/TF/xieshi/getPoetryFromPi3.py');

//const restart=require('child_process')

//restart.exec('pm2 restart all');

module.exports = (req, res) => {
    res.send(poetry0);
};