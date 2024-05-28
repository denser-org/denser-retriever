export function Faq() {
  return (
    <section className="faq">
      <div className="container relative max-w-4xl px-8 py-24 mx-auto mt-32 md:px-12 lg:px-32">
        <div className="absolute -left-16 -top-24 font-extrabold opacity-[0.05] font-heading text-[12rem]">
          FAQ
        </div>
        <div className="p-2 w-full rounded-xl border shadow-xs backdrop-blur md:w-[640px] dark:border-neutral-800">
          <div className="flex flex-col gap-6 p-10 text-base text-left border rounded-lg bg-gray-50 md:p-20 lg:col-span-2 text-neutral-400 dark:border-neutral-600 dark:bg-neutral-800 dark:text-neutral-50">
            <details name="faq" className="">
              <summary className="p-1 text-lg font-medium text-black rounded-sm outline-none cursor-pointer dark:text-white focus:ring-2 dark:focus:ring-neutral-600">
                Can I really use this for free?
              </summary>
              <p className="pt-4 text-base font-light leading-7 tracking-wide text-neutral-500 text-balance dark:text-neutral-50">
                Yes! This is a free and open-source project. You can use it for
                free, even for commercial purposes.
              </p>
            </details>
            <details name="faq">
              <summary className="p-1 text-lg font-medium text-black rounded-sm outline-none cursor-pointer dark:text-white focus:ring-2 dark:focus:ring-neutral-600">
                I&apos;ve found a bug or want to make a feature request
              </summary>
              <p className="pt-4 text-base font-light leading-7 tracking-wide text-neutral-500 text-balance dark:text-neutral-50">
                Great! Your best bet is to create an issue in the{" "}
                <a
                  href="https://github.com/denser-org/denser-retriever/issues"
                  className="underline underline-offset-4"
                  target="_blank"
                >
                  GitHub repository
                </a>
                . If you don&apos;t have a Github account, you can also email us
                at <code>support@denser.ai</code>.
              </p>
            </details>
          </div>
        </div>
      </div>
    </section>
  )
}
