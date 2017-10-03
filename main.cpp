#include <iostream>

#include <CL/sycl.hpp>
#include <SDL2/SDL.h>

namespace sycl = cl::sycl;

int main() {
    const size_t width = 32;
    const size_t height = 32;
    std::vector<cl_short> frameA(width * height, 0);
    std::vector<cl_short> frameB(width * height, 0);
    std::vector<cl_short>* currentFrame = &frameA;
    std::vector<cl_short>* prevFrame = &frameB;

    //TMP: Setup glider
    (*prevFrame)[5 * width + 5] = 1;
    (*prevFrame)[6 * width + 6] = 1;
    (*prevFrame)[7 * width + 6] = 1;
    (*prevFrame)[7 * width + 5] = 1;
    (*prevFrame)[7 * width + 4] = 1;

    for (int i = 1; i < height-1; ++i) {
        for (int j = 1; j < width-1; ++j) {
            std::cout << (*prevFrame)[i * width + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (int k = 0; k < 50; ++k) {
        
        {
            sycl::buffer<cl_short, 2> bufCurrent(currentFrame->data(), sycl::range<2>(width, height));
            sycl::buffer<cl_short, 2> bufPrev(prevFrame->data(), sycl::range<2>(width, height));
        
            sycl::queue commandQueue;
            commandQueue.submit([&] (sycl::handler& cgh) {
                auto ptrCurrent = bufCurrent.get_access<sycl::access::mode::write>(cgh);
                auto ptrPrev = bufPrev.get_access<sycl::access::mode::read>(cgh);

                cgh.parallel_for<class GameOfLifeLoop>(sycl::range<2>(width - 2, height - 2),
                                                       sycl::id<2>(1, 1),
                                                       [=](sycl::item<2> item) {
                    const size_t x = item.get(0);
                    const size_t y = item.get(1);

                    int neighbourCount = 0;
                    neighbourCount += ptrPrev[(y - 1)][x - 1];
                    neighbourCount += ptrPrev[(y - 1)][x    ];
                    neighbourCount += ptrPrev[(y - 1)][x + 1];
                    neighbourCount += ptrPrev[y][x - 1];
                    neighbourCount += ptrPrev[y][x + 1];
                    neighbourCount += ptrPrev[(y + 1)][x - 1];
                    neighbourCount += ptrPrev[(y + 1)][x    ];
                    neighbourCount += ptrPrev[(y + 1)][x + 1];
                
                    if (neighbourCount < 2 || neighbourCount > 3) { ptrCurrent[y][x] = 0; }
                    else if (neighbourCount == 3) { ptrCurrent[y][x] = 1; }
                    else { ptrCurrent[y][x] = ptrPrev[y][x]; }
                });
            });
        }

        for (int i = 1; i < height-1; ++i) {
            for (int j = 1; j < width-1; ++j) {
                std::cout << (*currentFrame)[i * width + j];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        auto tmp = currentFrame;
        currentFrame = prevFrame;
        prevFrame = tmp;
    }

    return 0;
}


