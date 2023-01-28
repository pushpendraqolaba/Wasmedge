import {Image} from 'image';
import * as std from 'std';
import {TensorflowLiteSession} from 'tensorflow_lite';

let img = new Image(__dirname + '/food.jpg');
let img_rgb = img.to_rgb().resize(192, 192);
let rgb_pix = img_rgb.pixels();

// /////Input for our text_encoder model//////////

// let phrase=[49406, 18376,  6765,   320,  4558,   525,   518,  3293,   267,
//   7997, 16157, 12066,   267,  6087,   525,  1486,  2631,   267,
//  27118,  3757,   267,  1313,  6999,  9529,   267,  3878, 15893,
//    267, 25602,   267, 17510,   267,  1215,  5799,   267, 22146,
//  12609,   267, 43040, 13024,   267,   275,   330,   267, 15636,
//   5857, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
//  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
//  49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
//  49407, 49407, 49407, 49407, 49407];

// let pos_ids=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
//   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
//   32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
//   48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
//   64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76];

/////// Ideal output for our text encoder should be 
// context=[[-0.67242926,  0.41616926, -0.96456504, ...,  0.01518078,
//   -0.56781065,  0.5645931 ],
//  [-0.79297835,  0.12064935, -1.3021824 , ...,  0.8641019 ,
//   -0.78513336, -0.29295063],
//  [-0.5372405 ,  0.64596254, -0.8686346 , ...,  0.49009895,
//   -0.72127444, -0.38098848],
//  ...,
//  [-0.4733906 ,  0.3820865 ,  0.2047154 , ...,  1.2678356 ,
//   -0.77972084, -1.0087771 ],
//  [-0.5594693 ,  0.30777645,  0.14900558, ...,  1.346043  ,
//   -0.78746164, -0.95352846],
//  [-0.65947145,  0.00828257,  0.27094707, ...,  1.5746298 ,
//   -1.1085671 , -1.0996103 ]];



let session = new TensorflowLiteSession(
    __dirname + '/lite-model_aiy_vision_classifier_food_V1_1.tflite');
session.add_input('input', rgb_pix);
session.run();
let output = session.get_output('MobilenetV1/Predictions/Softmax');

//////////  Text encoder session ////

// let session = new TensorflowLiteSession(
//   __dirname + '/lite-model_aiy_vision_classifier_food_V1_1.tflite');
// session.add_input('phrase', phrase);
// session.add_input('pos_ids',pos_ids);
// session.run();
// let context = session.get_output('MobilenetV1/Predictions/Softmax');


let output_view = new Uint8Array(output);
let max = 0;
let max_idx = 0;
for (var i in output_view) {
  let v = output_view[i];
  if (v > max) {
    max = v;
    max_idx = i;
  }
}
let label_file = std.open(__dirname + '/aiy_food_V1_labelmap.txt', 'r');
let label = '';
for (var i = 0; i <= max_idx; i++) {
  label = label_file.getline();
}
label_file.close();

print('label:');
print(label);
print('confidence:');
print(max / 255);
