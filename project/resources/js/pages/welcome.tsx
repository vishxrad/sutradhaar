import { type SharedData } from '@/types';
import { Head, Link, usePage } from '@inertiajs/react';

export default function Welcome() {
    const { auth } = usePage<SharedData>().props;
    function toggleMute() {
        const video = document.getElementById('myVideo') as HTMLVideoElement | null;
        const btn = document.getElementById('listenToggle') as HTMLButtonElement | null;

        if (video && btn) {
            video.muted = !video.muted;
            btn.textContent = video.muted ? "Sound On" : "Sound Off";
        }
    }

    return (
        <>
            <Head title="Welcome">
                <link rel="preconnect" href="https://fonts.bunny.net" />
                <link href="https://fonts.bunny.net/css?family=instrument-sans:400,500,600" rel="stylesheet" />
            </Head>
            <div className="flex min-h-screen flex-col items-center bg-[#FDFDFC] p-6 text-[#1b1b18] lg:justify-center lg:p-8 dark:bg-[#0a0a0a]">
                <header className="mb-6 w-full text-sm">
                    <nav className="flex items-center justify-end gap-4">
                        {auth.user ? (
                            <Link
                                href={route('dashboard')}
                                className="inline-block rounded-sm border border-[#19140035] px-5 py-1.5 text-sm leading-normal text-[#1b1b18] hover:border-[#1915014a] dark:border-[#3E3E3A] dark:text-[#EDEDEC] dark:hover:border-[#62605b]"
                            >
                                Dashboard
                            </Link>
                        ) : (
                            <>
                                <Link
                                    href={route('login')}
                                    className="inline-block rounded-sm border border-transparent px-5 py-1.5 text-sm leading-normal text-[#1b1b18] hover:border-[#19140035] dark:text-[#EDEDEC] dark:hover:border-[#3E3E3A]"
                                >
                                    Log in
                                </Link>
                                <Link
                                    href={route('register')}
                                    className="inline-block rounded-sm border border-[#19140035] px-5 py-1.5 text-sm leading-normal text-[#1b1b18] hover:border-[#1915014a] dark:border-[#3E3E3A] dark:text-[#EDEDEC] dark:hover:border-[#62605b]"
                                >
                                    Register
                                </Link>
                            </>
                        )}
                    </nav>
                </header>
                <div className="relative min-h-screen w-full bg-[#FDFDFC] text-[#1b1b18] dark:bg-[#0a0a0a]">
                    <main className="flex w-full flex-col items-center justify-center">
                        {/* <div
                            id="splashScreen"
                            className="w-full h-screen flex flex-col items-center justify-center px-4 sm:px-6 md:px-8 relative overflow-hidden"
                        > */}
                        <div className="w-full h-screen flex flex-col items-center justify-start pt-10 px-4 sm:px-6 md:px-8">


                            {/* Moving Dots Background */}
                            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                                {/* Large Soft Glowing Orbs */}
                                <div
                                    className="absolute w-96 h-96 bg-blue-400 rounded-full animate-soft-glow"
                                    style={{ left: '-10%', top: '-15%', animationDelay: '0s', animationDuration: '10s' }}
                                ></div>
                                <div
                                    className="absolute w-80 h-80 bg-purple-400 rounded-full animate-soft-glow"
                                    style={{ right: '-15%', top: '20%', animationDelay: '-3s', animationDuration: '12s' }}
                                ></div>
                                <div
                                    className="absolute w-72 h-72 bg-indigo-400 rounded-full animate-soft-glow"
                                    style={{ left: '20%', bottom: '-20%', animationDelay: '-6s', animationDuration: '9s' }}
                                ></div>

                                {/* Small Dots */}
                                {[
                                    { left: '10%', top: '20%', delay: '0.5s', duration: '3s' },
                                    { left: '80%', top: '30%', delay: '1s', duration: '2.5s' },
                                    { left: '60%', top: '70%', delay: '1.5s', duration: '4s' },
                                    { left: '30%', top: '80%', delay: '2s', duration: '3.5s' },
                                    { left: '90%', top: '60%', delay: '0.8s', duration: '2.8s' },
                                    { left: '5%', top: '50%', delay: '0.2s', duration: '4s', size: 'w-1 h-1', color: 'bg-purple-300/30' },
                                    { left: '95%', top: '10%', delay: '1.2s', duration: '3.2s', size: 'w-1.5 h-1.5', color: 'bg-indigo-300/30' },
                                    { left: '40%', top: '5%', delay: '1.8s', duration: '2.5s', size: 'w-1 h-1' },
                                    { left: '20%', top: '60%', delay: '0.3s', duration: '3.8s', size: 'w-2.5 h-2.5', color: 'bg-purple-300/30' },
                                    { left: '70%', top: '15%', delay: '1.7s', duration: '2.2s', size: 'w-1 h-1', color: 'bg-indigo-300/30' },
                                    { left: '50%', top: '90%', delay: '2.2s', duration: '3.3s' },
                                    { left: '5%', top: '5%', delay: '0.1s', duration: '4.5s', size: 'w-1.5 h-1.5', color: 'bg-purple-300/30' },
                                    { left: '85%', top: '85%', delay: '2.5s', duration: '3.7s', color: 'bg-indigo-300/20', size: 'w-1 h-1' },
                                    { left: '15%', top: '40%', delay: '0.9s', duration: '2.9s', color: 'bg-blue-300/20' }
                                ].map((dot, index) => (
                                    <div
                                        key={index}
                                        className={`absolute ${dot.size || 'w-2 h-2'} ${dot.color || 'bg-blue-300/30'} rounded-full animate-pulse`}
                                        style={{
                                            left: dot.left,
                                            top: dot.top,
                                            animationDelay: dot.delay,
                                            animationDuration: dot.duration
                                        }}
                                    />
                                ))}
                            </div>

                            {/* Main content */}
                            <div className="text-center space-y-6 max-w-2xl">
                                <h1
                                    className="text-5xl sm:text-6xl md:text-7xl lg:text-[180px] xl:text-[220px] font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-700 bg-clip-text text-transparent animate-fade-in z-10 leading-tight"
                                    style={{ animationDelay: '0.5s' }}
                                >
                                    सूत्रधार
                                </h1>

                                <p
                                    className="text-lg sm:text-xl md:text-3xl lg:text-5xl text-gray-700 leading-snug animate-fade-in font-semibold"
                                    style={{ animationDelay: '0.8s', fontFamily: 'inherit' }}
                                >
                                    Turn text into video,<br />
                                    in <span className="text-purple-600 font-bold">minutes.</span>
                                </p>

                                <p
                                    className="text-xs sm:text-sm md:text-base lg:text-lg text-gray-700 animate-fade-in"
                                    style={{ animationDelay: '1.1s' }}
                                >
                                    Transform Ideas → Script → Slides → Video
                                </p>


                                <div className="pt-8 animate-fade-in" style={{ animationDelay: '1.4s' }}>
                                    <nav className="flex items-center justify-center gap-6">
                                        <Link
                                            href={route('login')}
                                            className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold px-8 py-3 text-lg transition-all duration-300 transform hover:scale-105 hover:shadow-xl shadow-md rounded-md"
                                        >
                                            Log In
                                        </Link>
                                        <Link
                                            href={route('register')}
                                            className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white font-semibold px-8 py-3 text-lg transition-all duration-300 transform hover:scale-105 hover:shadow-xl shadow-md rounded-md"
                                        >
                                            Sign Up
                                        </Link>
                                    </nav>
                                </div>

                            </div>



                            {/* <div className="relative w-[80%] h-[800px] mx-auto mt-8 rounded-xl overflow-hidden shadow-xl border-4 border-transparent bg-clip-padding">
                                <div className="w-full h-full bg-black rounded-lg overflow-hidden">
                                    <video
                                        id="myVideo"
                                        src="generated_final_videos/script_1748465992/final_presentation_script_1748465992.mp4"
                                        controls
                                        autoPlay
                                        muted
                                        className="w-full h-full object-cover"
                                    ></video>

                                    {/* Listen Button Overlay */}
                            {/* <button
                                        id="listenToggle"
                                        className="absolute top-4 right-4 bg-white/80 text-black font-semibold px-4 py-2 rounded-md shadow-md transition border"
                                        onClick={toggleMute}
                                    >
                                        Sound On
                                    </button>
                                </div>
                            </div> */}
                        </div>
                    </main>
                    <div className="hidden h-14.5 lg:block"></div>
                </div>
            </div>
        </>
    );
}
