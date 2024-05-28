export function Pricing() {
  return (
    <section>
      <div className="container relative items-center w-full mx-auto mt-40 md:px-12 lg:px-16">
        <div className="absolute -left-16 font-extrabold -top-[9.5rem] opacity-[0.05] font-heading text-[12rem]">
          Pricing
        </div>
        <div>
          <div className="relative p-3 space-y-12 overflow-hidden lg:grid lg:grid-cols-2 lg:gap-x-8 lg:p-10 lg:space-y-0">
            <div className="p-2 border shadow-xs rounded-xl backdrop-blur dark:border-neutral-800">
              <div className="relative flex flex-col p-8 border rounded-lg bg-neutral-700 dark:border-neutral-600">
                <div className="relative flex-1">
                  <h3 className="text-3xl font-semibold text-white font-display">
                    Silicon Sailor
                  </h3>
                  <p className="flex flex-col items-center my-8 text-white xl:flex-row xl:justify-center xl:items-baseline">
                    <span className="flex items-baseline">
                      <span className="text-4xl font-extrabold tracking-tight">
                        $
                      </span>
                      <span className="text-6xl font-extrabold tracking-tight md:text-8xl lg:text-7xl xl:text-8xl">
                        0.00
                      </span>
                    </span>
                    <span className="ml-1 font-semibold lg:text-lg xl:text-xl text-md">
                      / month
                    </span>
                  </p>
                  <ul role="list" className="pt-6 mt-6 space-y-6">
                    <li className="flex">
                      <div className="inline-flex items-center w-6 h-6 bg-white rounded-full">
                        <svg
                          className="flex-shrink-0 w-4 h-4 mx-auto text-neutral-600"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          aria-hidden="true"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M5 13l4 4L19 7"
                          ></path>
                        </svg>
                      </div>
                      <span className="ml-3 text-white">
                        Unlimited Bookmarks
                      </span>
                    </li>
                    <li className="flex">
                      <div className="inline-flex items-center w-6 h-6 bg-white rounded-full min-w-6">
                        <svg
                          className="flex-shrink-0 w-4 h-4 mx-auto text-neutral-600"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          aria-hidden="true"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M5 13l4 4L19 7"
                          ></path>
                        </svg>
                      </div>
                      <span className="ml-3 text-white">
                        Unlimited RSS Feeds
                      </span>
                    </li>
                    <li className="flex">
                      <div className="inline-flex items-center w-6 h-6 bg-white rounded-full min-w-6">
                        <svg
                          className="flex-shrink-0 w-4 h-4 mx-auto text-neutral-600"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          aria-hidden="true"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M5 13l4 4L19 7"
                          ></path>
                        </svg>
                      </div>
                      <span className="ml-3 text-left text-white">
                        Unlimited Text-to-Speech and Summarizations
                      </span>
                    </li>
                    <li className="flex">
                      <div className="inline-flex items-center w-6 h-6 bg-white rounded-full min-w-6">
                        <svg
                          className="flex-shrink-0 w-4 h-4 mx-auto text-neutral-600"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          aria-hidden="true"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M5 13l4 4L19 7"
                          ></path>
                        </svg>
                      </div>
                      <span className="ml-3 text-white">Community Support</span>
                    </li>
                  </ul>
                </div>
                <div className="z-50 mt-6">
                  <a
                    href="/docs/self-hosting"
                    type="highlight"
                    className="flex justify-center items-center py-3.5 px-10 w-full text-xl font-bold bg-white rounded-md border-2 border-white transition duration-500 ease-in-out transform focus:ring-2 focus:ring-offset-2 focus:outline-none group shadow-xs font-display text-neutral-600 focus:ring-neutral-400 focus:ring-offset-neutral-700"
                  >
                    Selfhost
                    <div className="hidden lg:block w-0 translate-x-[100%] pl-0 opacity-0 transition-all duration-200 lg:group-hover:w-5 lg:group-hover:translate-x-0 lg:group-hover:pl-4 lg:group-hover:opacity-100">
                      <svg
                        className="size-6"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 256 256"
                      >
                        <rect width="256" height="256" fill="none" />
                        <ellipse
                          cx="128"
                          cy="80"
                          rx="88"
                          ry="48"
                          fill="none"
                          stroke="currentColor"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="16"
                        />
                        <path
                          d="M40,80v48c0,26.51,39.4,48,88,48s88-21.49,88-48V80"
                          fill="none"
                          stroke="currentColor"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="16"
                        />
                        <path
                          d="M40,128v48c0,26.51,39.4,48,88,48s88-21.49,88-48V128"
                          fill="none"
                          stroke="currentColor"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="16"
                        />
                      </svg>
                    </div>
                  </a>
                </div>
              </div>
            </div>
            <div className="p-2 border shadow-xs rounded-xl backdrop-blur dark:border-neutral-800">
              <div className="relative flex flex-col p-8 bg-gray-200 border rounded-lg dark:border-neutral-600">
                <div className="flex-1">
                  <h3 className="text-3xl font-semibold text-neutral-600 font-display">
                    Cumulus Cruiser
                  </h3>
                  <p className="flex flex-col items-center my-8 xl:flex-row xl:justify-center xl:items-baseline text-neutral-600">
                    <span className="flex items-baseline">
                      <span className="text-4xl font-extrabold tracking-tight">
                        $
                      </span>
                      <span className="text-6xl font-extrabold tracking-tight md:text-8xl lg:text-7xl xl:text-8xl">
                        0.00
                      </span>
                    </span>
                    <span className="ml-1 font-semibold lg:text-lg xl:text-xl text-md">
                      / month
                    </span>
                  </p>
                  <ul role="list" className="pt-6 mt-6 space-y-6">
                    <li className="flex">
                      <div className="inline-flex items-center w-6 h-6 rounded-full bg-neutral-600 min-w-6">
                        <svg
                          className="flex-shrink-0 w-4 h-4 mx-auto text-white"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          aria-hidden="true"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M5 13l4 4L19 7"
                          ></path>
                        </svg>
                      </div>
                      <span className="ml-3 text-neutral-600">
                        Unlimited Bookmarks
                      </span>
                    </li>
                    <li className="flex">
                      <div className="inline-flex items-center w-6 h-6 rounded-full bg-neutral-600 min-w-6">
                        <svg
                          className="flex-shrink-0 w-4 h-4 mx-auto text-white"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          aria-hidden="true"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M5 13l4 4L19 7"
                          ></path>
                        </svg>
                      </div>
                      <span className="ml-3 text-neutral-600">
                        Unlimited RSS Feeds
                      </span>
                    </li>
                    <li className="flex">
                      <div className="inline-flex items-center w-6 h-6 rounded-full bg-neutral-600 min-w-6">
                        <svg
                          className="flex-shrink-0 w-4 h-4 mx-auto text-white"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          aria-hidden="true"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M5 13l4 4L19 7"
                          ></path>
                        </svg>
                      </div>
                      <span className="ml-3 text-left text-neutral-600">
                        Unlimited Text-to-Speech and Summarizations
                      </span>
                    </li>
                    <li className="flex">
                      <div className="inline-flex items-center w-6 h-6 rounded-full bg-neutral-600 min-w-6">
                        <svg
                          className="flex-shrink-0 w-4 h-4 mx-auto text-white"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          aria-hidden="true"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M5 13l4 4L19 7"
                          ></path>
                        </svg>
                      </div>
                      <span className="ml-3 text-neutral-600">
                        Community Support
                      </span>
                    </li>
                  </ul>
                </div>
                <div className="mt-6">
                  <a
                    href="https://dev.denser.ai/?utm_source=docs&utm_medium=cta&utm_campaign=pricing"
                    target="_blank"
                    type="highlight"
                    className="flex justify-center items-center py-3.5 px-10 w-full text-xl font-bold text-white rounded-md border-2 transition duration-500 ease-in-out transform focus:ring-2 focus:ring-offset-2 focus:outline-none group border-neutral-700 bg-neutral-700 shadow-xs font-display focus:ring-neutral-200"
                  >
                    Cloud
                    <div className="hidden lg:block w-0 translate-x-[100%] pl-0 opacity-0 text-white transition-all duration-200 lg:group-hover:w-5 lg:group-hover:translate-x-0 lg:group-hover:pl-4 lg:group-hover:opacity-100">
                      <svg
                        className="size-6"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 256 256"
                      >
                        <rect width="256" height="256" fill="none" />
                        <path
                          d="M80,128a80,80,0,1,1,80,80H72A56,56,0,1,1,85.92,97.74"
                          fill="none"
                          stroke="currentColor"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="16"
                        />
                      </svg>
                    </div>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
