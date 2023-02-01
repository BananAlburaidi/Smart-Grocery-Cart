/*
This is the javascript file. It is used to run the web app for smart shopping cart.
To run the detect.py file, we have used python-shell library.

To start the web app and detection, only write:

//*         py -m venv .venv
//*         .venv/Scripts/activate
//*         npm install
//*         npm start


NOTE: Install Python and NodeJS in your computer
*/


import express from "express"
import PythonShellLibrary from 'python-shell'
import cors from "cors"

let {PythonShell} = PythonShellLibrary;
// ========================================================================================================
let py_script = new PythonShell('./detect.py', {
    // The '-u' tells Python to flush every time
    pythonOptions: ['-u'],
    args: ['--weights', 'best.pt', '--source', '0', '--exefrom', 'node']    //* just change '0' to 'https://192.168.X.X/capture' the ip address of ESP32
    // python detect.py --weights .\best.pt --source 0 --exefrom node 
});

// ========================================================================================================


const app = express();
app.use(cors()) // middleware

const PORT = 8000


app.get("/", (req, res) =>{
    py_script.on('message', function(result){

        setTimeout(() => {
            res.write(result)
            res.end("")

        }, 200);    // sending response after every 200 ms. this is basically a time delay
    })
    
})


app.listen(PORT, () => console.log(`Express server currently running on port ${PORT}`));
