//
// Created by stijn on 7/24/23.
//

#include <iostream>
#include <string>
#include "typing.h"
#include "motif.h"
#include "miner.h"
#include "sax.h"

int main()
{
    TimeSeriesDB data { { 87.614, 70.338, 82.678, 117.23, 90.082, 60.466, 72.806, 29.616, 23.445999999999998, 0.0, 81.444, 3.702, 37.019999999999996, 85.146, 114.762, 69.104, 104.89, 101.188, 104.89, 6.17, 25.914, 118.464, 122.166, 19.744, 99.954, 22.212, 67.87, 111.06, 88.848, 54.296, 53.062, 88.848, 101.188, 43.19, 20.978, 9.872, 95.018, 17.276, 77.742, 41.956, 4.936, 117.23, 23.445999999999998, 4.936, 33.318, 97.486, 18.509999999999998, 12.34, 104.89, 61.7, 92.55, 83.912, 120.932, 120.932, 9.872, 103.656, 77.742, 17.276, 86.38, 85.146, 45.658, 11.106, 56.763999999999996, 97.486, 66.636, 80.21, 83.912, 75.274, 64.168, 34.552, 108.592, 93.78399999999999, 76.508, 113.52799999999999, 87.614, 20.978, 53.062, 74.03999999999999, 59.232, 50.594, 71.572, 57.998, 93.78399999999999, 109.826, 43.19, 120.932, 40.722, 118.464, 99.954, 114.762, 114.762, 3.702, 85.146, 81.444, 54.296, 99.954, 108.592, 12.34, 39.488, 115.996, },
                        { 25.914, 59.232, 45.658, 80.21, 91.316, 41.956, 69.104, 48.126, 90.082, 60.466, 96.252, 14.808, 115.996, 16.042, 18.509999999999998, 112.294, 40.722, 25.914, 37.019999999999996, 59.232, 103.656, 14.808, 55.53, 88.848, 6.17, 87.614, 4.936, 1.234, 25.914, 18.509999999999998, 66.636, 115.996, 19.744, 60.466, 82.678, 11.106, 117.23, 106.124, 75.274, 14.808, 111.06, 67.87, 40.722, 6.17, 48.126, 107.358, 7.404, 13.574, 59.232, 4.936, 28.381999999999998, 98.72, 54.296, 92.55, 106.124, 46.891999999999996, 30.85, 76.508, 20.978, 64.168, 101.188, 99.954, 109.826, 20.978, 30.85, 1.234, 55.53, 80.21, 54.296, 120.932, 81.444, 14.808, 9.872, 43.19, 74.03999999999999, 25.914, 32.084, 109.826, 34.552, 8.638, 114.762, 32.084, 98.72, 53.062, 43.19, 9.872, 109.826, 76.508, 111.06, 69.104, 64.168, 67.87, 35.786, 78.976, 87.614, 83.912, 91.316, 3.702, 69.104, 101.188, },
                        { 2.468, 23.445999999999998, 62.934, 69.104, 82.678, 32.084, 61.7, 119.698, 97.486, 104.89, 81.444, 1.234, 115.996, 99.954, 119.698, 93.78399999999999, 17.276, 64.168, 87.614, 120.932, 46.891999999999996, 102.422, 30.85, 3.702, 117.23, 66.636, 66.636, 43.19, 85.146, 50.594, 75.274, 114.762, 96.252, 91.316, 13.574, 34.552, 33.318, 48.126, 17.276, 120.932, 43.19, 4.936, 2.468, 34.552, 7.404, 57.998, 97.486, 24.68, 32.084, 56.763999999999996, 88.848, 16.042, 104.89, 6.17, 78.976, 104.89, 66.636, 61.7, 111.06, 37.019999999999996, 109.826, 27.148, 66.636, 41.956, 35.786, 82.678, 101.188, 45.658, 106.124, 9.872, 75.274, 107.358, 18.509999999999998, 39.488, 115.996, 112.294, 37.019999999999996, 2.468, 49.36, 77.742, 65.402, 107.358, 118.464, 115.996, 2.468, 29.616, 82.678, 98.72, 32.084, 46.891999999999996, 25.914, 43.19, 104.89, 51.828, 60.466, 9.872, 14.808, 104.89, 20.978, 96.252, },
                        { 92.55, 27.148, 90.082, 71.572, 122.166, 96.252, 14.808, 55.53, 107.358, 43.19, 72.806, 106.124, 72.806, 6.17, 6.17, 106.124, 71.572, 119.698, 119.698, 101.188, 98.72, 102.422, 43.19, 1.234, 20.978, 61.7, 2.468, 3.702, 59.232, 91.316, 96.252, 16.042, 29.616, 101.188, 99.954, 59.232, 46.891999999999996, 27.148, 54.296, 108.592, 113.52799999999999, 11.106, 29.616, 38.254, 40.722, 20.978, 119.698, 77.742, 111.06, 90.082, 28.381999999999998, 80.21, 38.254, 0.0, 81.444, 112.294, 43.19, 12.34, 85.146, 12.34, 7.404, 13.574, 45.658, 66.636, 20.978, 8.638, 80.21, 16.042, 20.978, 20.978, 106.124, 93.78399999999999, 40.722, 77.742, 32.084, 19.744, 107.358, 50.594, 0.0, 50.594, 81.444, 113.52799999999999, 48.126, 77.742, 87.614, 25.914, 13.574, 117.23, 13.574, 71.572, 87.614, 119.698, 96.252, 28.381999999999998, 117.23, 85.146, 101.188, 82.678, 91.316, 75.274, },
                        { 11.106, 93.78399999999999, 83.912, 103.656, 87.614, 45.658, 62.934, 55.53, 29.616, 102.422, 39.488, 83.912, 90.082, 74.03999999999999, 71.572, 24.68, 80.21, 117.23, 107.358, 37.019999999999996, 118.464, 93.78399999999999, 53.062, 6.17, 48.126, 70.338, 20.978, 22.212, 57.998, 75.274, 104.89, 40.722, 80.21, 72.806, 0.0, 11.106, 32.084, 111.06, 122.166, 107.358, 25.914, 38.254, 83.912, 75.274, 75.274, 103.656, 98.72, 62.934, 44.424, 122.166, 19.744, 71.572, 101.188, 93.78399999999999, 1.234, 25.914, 60.466, 7.404, 70.338, 107.358, 28.381999999999998, 97.486, 112.294, 45.658, 99.954, 93.78399999999999, 29.616, 118.464, 70.338, 101.188, 28.381999999999998, 7.404, 41.956, 17.276, 46.891999999999996, 64.168, 95.018, 76.508, 57.998, 69.104, 103.656, 119.698, 29.616, 115.996, 107.358, 18.509999999999998, 46.891999999999996, 96.252, 61.7, 30.85, 75.274, 108.592, 65.402, 34.552, 71.572, 54.296, 29.616, 91.316, 29.616, 71.572, },
                        { 508, 82.678, 78.976, 106.124, 29.616, 7.404, 30.85, 12.34, 49.36, 22.212, 14.808, 33.318, 43.19, 34.552, 45.658, 40.722, 1.234, 93.78399999999999, 41.956, 33.318, 80.21, 17.276, 34.552, 33.318, 4.936, 48.126, 41.956, 46.891999999999996, 61.7, 91.316, 104.89, 67.87, 20.978, 34.552, 107.358, 12.34, 17.276, 75.274, 24.68, 65.402, 28.381999999999998, 40.722, 112.294, 11.106, 49.36, 9.872, 111.06, 114.762, 65.402, 32.084, 83.912, 9.872, 54.296, 64.168, 71.572, 25.914, 70.338, 66.636, 46.891999999999996, 37.019999999999996, 19.744, 45.658, 3.702, 115.996, 64.168, 22.212, 8.638, 75.274, 27.148, 108.592, 96.252, 71.572, 96.252, 113.52799999999999, 16.042, 51.828, 34.552, 29.616, 108.592, 53.062, 119.698, 20.978, 51.828, 69.104, 3.702, 62.934, 61.7, 11.106, 93.78399999999999, 39.488, 17.276, 119.698, 27.148, 119.698, 107.358, 115.996, 35.786, 45.658, 70.338, 70.338, },
                        { -0.10730004710876988, 1.8439291451452031, -1.353396746395181, 0.14322551930584373, 0.1759981976854589, 1.3389808987817684, -0.40967767798347526, 1.1826623395931826, -1.3537552446411383, 0.2475856278212355, -0.31431638294219655, 1.2879594533439802, -0.04020922136660457, -1.1319234801545106, 0.9609436976255793, -1.3176254493605528, -0.2648354003102859, 0.27243085729357386, 1.0702742914700003, -0.3348238316643956, -1.019762264091206, 1.344012166653886, -1.328770843475085, 1.5009328072064356, -1.01189540235773, -0.9489515340482074, -0.8161114447370692, -0.8747366876835542, 1.2441147908901125, 1.2972218146487948, -0.8054944332040592, -0.9921631198227945, 0.5154776038817603 }
    };
    sax(data, 2, 4);
    Miner miner {0.5, 2, 3};
    miner.mine(data);
}